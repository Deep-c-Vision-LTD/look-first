"""
Train the text-conditioned ASPP-FiLM readout on RefCOCO-Gaze (free-text referring
expressions + human gaze, MIT licence) for the RA-L glimpse-planning letter.

Two things are measured at every checkpoint:
  1. map accuracy   -- CC for matched / unconditional / mismatched expression
  2. search cost    -- mean glimpses-to-target when the predicted density orders
                       the crops, against raster, centre-out, and a
                       leave-one-observer-out human-gaze oracle (the ceiling).

Splits: the official RefCOCO-Gaze val is only 92 stimuli, so the 1,799 official
train stimuli are split 1,500 / 299 (seed 0) and BOTH validation sets are
reported -- internal-299 for stability, official-92 for comparability with the
dataset's own papers.

  cd saliency_research && PYTHONPATH=. .venv/bin/python train_refcoco_gaze.py
"""
import os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_TRAIN = f"{ROOT}/data/refcoco_gaze/proc/intents_train.json"
PROC_VAL = f"{ROOT}/data/refcoco_gaze/proc/intents_val.json"
RAW_TRAIN = f"{ROOT}/data/refcoco_gaze/refcocogaze_train.json"
RAW_VAL = f"{ROOT}/data/refcoco_gaze/refcocogaze_val.json"
OUT_CKPT = f"{ROOT}/checkpoints/refcoco_gaze_v1.pt"
OUT_JSON = f"{ROOT}/checkpoints/refcoco_gaze_v1_results.json"

SIZE, EVAL, LOSSG = 448, 224, 64
EPOCHS, BS, LR = 60, 16, 3e-4
SIGMAS = [0, 1, 2, 3, 4, 6]
GRIDS = (4, 6)
DISP_W, DISP_H = 1680, 1050
N_INTERNAL_VAL = 299
IM_M = np.array([0.485, 0.456, 0.406], np.float32)
IM_S = np.array([0.229, 0.224, 0.225], np.float32)


def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


# ---------------- glimpse-planning metric (normalised [0,1]^2 space) ----------------
def tile_of(xn, yn, G):
    return min(int(yn * G), G - 1) * G + min(int(xn * G), G - 1)


def order_from_map(pred, G):
    """rank tiles by summed predicted density (pred is a square EVALxEVAL map)"""
    n = pred.shape[0]
    w = np.zeros(G * G)
    for r in range(G):
        for c in range(G):
            w[r * G + c] = pred[int(r * n / G):int((r + 1) * n / G),
                                int(c * n / G):int((c + 1) * n / G)].sum()
    return list(np.argsort(-w))


def order_raster(G):
    return list(range(G * G))


def order_centre_out(G):
    cs = [((c + .5) / G - .5) ** 2 + ((r + .5) / G - .5) ** 2
          for r in range(G) for c in range(G)]
    return list(np.argsort(cs))


def order_human(obs, G, exclude=None):
    """pooled dwell-time density per tile, leave-one-observer-out"""
    w = np.zeros(G * G)
    for i, t in enumerate(obs):
        if exclude is not None and i == exclude:
            continue
        for x, y, d in zip(t["FIX_X"], t["FIX_Y"], t["FIX_DURATION"]):
            w[tile_of(x / DISP_W, y / DISP_H, G)] += d
    return list(np.argsort(-w))


def glimpses(order, bbox, G):
    cx = (bbox[0] + bbox[2] / 2) / DISP_W
    cy = (bbox[1] + bbox[3] / 2) / DISP_H
    tgt = tile_of(cx, cy, G)
    return order.index(tgt) + 1


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {dev}", flush=True)
    torch.manual_seed(0); np.random.seed(0)

    model = IntentASPP().to(dev)

    all_train = json.load(open(PROC_TRAIN))
    official_val = json.load(open(PROC_VAL))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(all_train))
    internal_val = [all_train[i] for i in perm[:N_INTERNAL_VAL]]
    train = [all_train[i] for i in perm[N_INTERNAL_VAL:]]
    print(f"train {len(train)} | internal-val {len(internal_val)} | official-val {len(official_val)}",
          flush=True)

    # bbox + per-observer fixations, keyed by image filename
    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)):
            meta[t["IMAGEFILE"]].append(t)
    print(f"raw metadata for {len(meta)} stimuli", flush=True)

    def stem(e):
        return e["image"].split("/")[-1]

    # ---- text embeddings for every expression (free text, ~1.8k unique) ----
    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official_val})
    embs = []
    with torch.no_grad():
        for i in range(0, len(exprs), 256):
            embs.append(model.text_embed(exprs[i:i + 256], dev).cpu())
    temb = torch.cat(embs).to(dev)
    t_idx = {s: i for i, s in enumerate(exprs)}
    zero_cond = torch.zeros(1, temb.shape[1], device=dev)   # FiLM zero-init => unconditional
    print(f"{len(exprs)} expressions embedded", flush=True)

    # ---- cache frozen backbone features once ----
    paths = sorted({e["image"] for e in all_train} | {e["image"] for e in official_val})
    feats = {}
    with torch.no_grad():
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            feats[p] = model.visual_features(x)[0].half().cpu()
            if (i + 1) % 300 == 0:
                print(f"  cached {i+1}/{len(paths)}", flush=True)

    gtl = [torch.tensor(load_map(e["gt"], LOSSG), dtype=torch.float32) for e in train]

    params = [p for n, p in model.named_parameters()
              if not (n.startswith("vit.") or n.startswith("text_enc."))]
    for p in params:
        p.requires_grad_(True)
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    @torch.no_grad()
    def predict(entries, cond_mode="matched"):
        model.eval()
        out = []
        for i, e in enumerate(entries):
            f = feats[e["image"]].float()[None].to(dev)
            if cond_mode == "matched":
                c = temb[t_idx[e["intent"]]][None]
            elif cond_mode == "uncond":
                c = zero_cond
            else:   # mismatched: another stimulus's expression
                c = temb[t_idx[entries[(i + 1) % len(entries)]["intent"]]][None]
            s = model.readout_forward(f, c)
            out.append(torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                       align_corners=False))[0, 0].cpu().numpy().astype(np.float64))
        return out

    @torch.no_grad()
    def pick_sigma(n_sub=200):
        sub = np.random.default_rng(0).choice(len(train), min(n_sub, len(train)), replace=False)
        ents = [train[j] for j in sub]
        pr = predict(ents)
        gts = [load_map(e["gt"], EVAL) for e in ents]
        best, bcc = 0, -2
        for s in SIGMAS:
            m = float(np.mean([cc_np(gaussian_filter(p, s) if s else p, g) for p, g in zip(pr, gts)]))
            if m > bcc:
                bcc, best = m, s
        return best

    def evaluate(entries, tag, sigma):
        gte = [load_map(e["gt"], EVAL) for e in entries]
        rep = {}
        for mode in ("matched", "uncond", "mismatched"):
            pr = [gaussian_filter(p, sigma) if sigma else p for p in predict(entries, mode)]
            rep[f"cc_{mode}"] = float(np.mean([cc_np(p, g) for p, g in zip(pr, gte)]))
            if mode == "matched":
                matched_pr = pr
        # glimpse planning
        for G in GRIDS:
            mm, rr, cc_, hh = [], [], [], []
            for e, p in zip(entries, matched_pr):
                obs = meta.get(stem(e), [])
                if not obs:
                    continue
                bb = obs[0]["BBOX"]
                mm.append(glimpses(order_from_map(p, G), bb, G))
                rr.append(glimpses(order_raster(G), bb, G))
                cc_.append(glimpses(order_centre_out(G), bb, G))
                hh.append(float(np.mean([glimpses(order_human(obs, G, exclude=i), bb, G)
                                         for i in range(len(obs))])))
            rep[f"glimpse_{G}x{G}"] = {
                "model": float(np.mean(mm)), "raster": float(np.mean(rr)),
                "centre_out": float(np.mean(cc_)), "human_oracle": float(np.mean(hh)),
                "model_succ@1": float(np.mean(np.array(mm) <= 1)),
                "model_succ@3": float(np.mean(np.array(mm) <= 3)),
                "saving_vs_raster": float(1 - np.mean(mm) / np.mean(rr)),
                "n": len(mm)}
        print(f"  [{tag}] CC matched {rep['cc_matched']:.4f} | uncond {rep['cc_uncond']:.4f} "
              f"| mismatched {rep['cc_mismatched']:.4f}", flush=True)
        for G in GRIDS:
            d = rep[f"glimpse_{G}x{G}"]
            print(f"     {G}x{G} glimpses: model {d['model']:.2f} vs raster {d['raster']:.2f} "
                  f"/ centre {d['centre_out']:.2f} / human {d['human_oracle']:.2f}  "
                  f"({100*d['saving_vs_raster']:.0f}% saving, succ@3 {100*d['model_succ@3']:.0f}%)",
                  flush=True)
        return rep

    history, best = [], -1
    order = np.arange(len(train))
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order)
        for j0 in range(0, len(order), BS):
            ids = order[j0:j0 + BS]
            f = torch.stack([feats[train[j]["image"]].float() for j in ids]).to(dev)
            c = temb[[t_idx[train[j]["intent"]] for j in ids]]
            g = torch.stack([gtl[j] for j in ids]).to(dev)
            s = model.readout_forward(f, c)
            s = F.interpolate(s, size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            ps = torch.sigmoid(s)
            pc, gc = ps - ps.mean((1, 2), keepdim=True), g - g.mean((1, 2), keepdim=True)
            ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc**2).sum((1, 2)) * (gc**2).sum((1, 2))) + 1e-8)
            loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            sig = pick_sigma()
            print(f"epoch {ep+1} (sigma={sig})", flush=True)
            rep_i = evaluate(internal_val, "internal-299", sig)
            rep_o = evaluate(official_val, "official-92", sig)
            history.append({"epoch": ep + 1, "sigma": sig,
                            "internal": rep_i, "official": rep_o})
            if rep_i["cc_matched"] > best:
                best = rep_i["cc_matched"]
                torch.save({k: v for k, v in model.state_dict().items()
                            if not (k.startswith("vit.") or k.startswith("text_enc."))}, OUT_CKPT)
            json.dump({"history": history, "best_internal_cc_matched": best},
                      open(OUT_JSON, "w"), indent=2)

    print(f"\nBEST internal-val matched CC = {best:.4f}  -> {OUT_CKPT}", flush=True)


if __name__ == "__main__":
    main()
