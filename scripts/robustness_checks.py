"""
Two robustness checks for the RA-L letter, both local (no GPU beyond the cached
readout forward passes).

(1) Grid robustness: is the glimpse-cost advantage an artefact of one axis-aligned
    grid? Re-score under jittered grid origins and under half-tile-overlapping
    windows. If the advantage survives, the static-grid choice is not load-bearing.

(2) Paraphrase robustness: does the model respond to the MEANING of the expression
    or to memorised surface strings? Re-score with reworded expressions that keep
    the referent (and its spatial cue) but change the wording, and check that
    search cost holds. This is the RefCOCO-Gaze analogue of the SPL paraphrase test.

  cd saliency_research && PYTHONPATH=. .venv/bin/python robustness_checks.py
"""
import json, re
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, DISP_W, DISP_H, load_map)

OUT_JSON = f"{ROOT}/checkpoints/refcoco_robustness.json"
G = 8


def paraphrase(expr):
    """meaning-preserving rewordings; returns a list of variants (never the original)."""
    e = expr.strip()
    outs = []
    # spatial-cue rewrites
    subs = [(r"\bon (the )?right\b", "on the right side"),
            (r"\bon (the )?left\b", "on the left side"),
            (r"\btop right\b", "upper right"),
            (r"\btop left\b", "upper left"),
            (r"\bbottom right\b", "lower right"),
            (r"\bin front\b", "at the front"),
            (r"\bfar left\b", "leftmost"),
            (r"\bfar right\b", "rightmost")]
    for pat, rep in subs:
        v = re.sub(pat, rep, e)
        if v != e:
            outs.append(v)
    # generic determiner prefix (meaning-preserving)
    outs.append("the " + e if not e.startswith("the ") else e[4:])
    # "find the X" framing
    outs.append("find the " + e)
    return list(dict.fromkeys(outs))[:3]   # up to 3 variants


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = IntentASPP().to(dev)
    model.load_state_dict(torch.load(OUT_CKPT, map_location="cpu"), strict=False)
    model.eval()

    official = json.load(open(PROC_VAL))
    all_train = json.load(open(PROC_TRAIN))
    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official})
    with torch.no_grad():
        temb = torch.cat([model.text_embed(exprs[i:i+256], dev).cpu()
                          for i in range(0, len(exprs), 256)]).to(dev)
    t_idx = {s: i for i, s in enumerate(exprs)}

    meta = defaultdict(list)
    for t in json.load(open(RAW_VAL)):
        meta[t["IMAGEFILE"]].append(t)

    # cache one prediction map per stimulus (matched expression)
    feats, preds, boxes = {}, {}, {}
    with torch.no_grad():
        for e in official:
            stem = e["image"].split("/")[-1]
            if stem not in meta:
                continue
            im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            feats[stem] = model.visual_features(x)
            s = model.readout_forward(feats[stem], temb[t_idx[e["intent"]]][None])
            preds[stem] = gaussian_filter(torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL),
                mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float64), 2)
            boxes[stem] = meta[stem][0]["BBOX"]

    def cost(pred, bb, gx, gy, ox=0.0, oy=0.0, overlap=1.0):
        """glimpses-to-target under a grid with fractional origin offset (ox,oy) and
        a window size = overlap x tile (overlap>1 => overlapping windows)."""
        tw, th = 1.0 / gx, 1.0 / gy
        n = pred.shape[0]
        centres = [((c + 0.5) * tw + ox * tw, (r + 0.5) * th + oy * th)
                   for r in range(gy) for c in range(gx)]
        w = tw * overlap, th * overlap
        scores = []
        for cx, cy in centres:
            x0, x1 = max(0, cx - w[0]/2), min(1, cx + w[0]/2)
            y0, y1 = max(0, cy - w[1]/2), min(1, cy + w[1]/2)
            scores.append(pred[int(y0*n):max(int(y0*n)+1, int(y1*n)),
                               int(x0*n):max(int(x0*n)+1, int(x1*n))].sum())
        order = list(np.argsort(-np.array(scores)))
        tx = (bb[0] + bb[2]/2) / DISP_W
        ty = (bb[1] + bb[3]/2) / DISP_H
        # target tile = nearest centre
        d = [ (cx-tx)**2 + (cy-ty)**2 for cx, cy in centres ]
        tgt = int(np.argmin(d))
        return order.index(tgt) + 1

    out = {}

    # ---- (1) grid robustness ----
    base = np.mean([cost(preds[s], boxes[s], G, G) for s in preds])
    jit = []
    rng = np.random.default_rng(0)
    for _ in range(8):
        ox, oy = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)
        jit.append(np.mean([cost(preds[s], boxes[s], G, G, ox, oy) for s in preds]))
    ov = np.mean([cost(preds[s], boxes[s], G, G, overlap=1.5) for s in preds])
    out["grid_robustness"] = {
        "base_8x8": float(base),
        "jittered_origin_mean": float(np.mean(jit)),
        "jittered_origin_std": float(np.std(jit)),
        "overlap_1.5x": float(ov),
        "n": len(preds)}
    print(f"(1) grid robustness (mean glimpses, n={len(preds)}):")
    print(f"    aligned 8x8         {base:.2f}")
    print(f"    jittered origin     {np.mean(jit):.2f} +/- {np.std(jit):.2f}  (8 random offsets)")
    print(f"    1.5x overlap window {ov:.2f}")

    # ---- (2) paraphrase robustness ----
    orig_costs, para_costs, n_para = [], [], 0
    with torch.no_grad():
        for e in official:
            stem = e["image"].split("/")[-1]
            if stem not in preds:
                continue
            variants = paraphrase(e["intent"])
            if not variants:
                continue
            orig_costs.append(cost(preds[stem], boxes[stem], G, G))
            vs = []
            for v in variants:
                c = model.text_embed([v], dev)
                s = model.readout_forward(feats[stem], c)
                p = gaussian_filter(torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL),
                    mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float64), 2)
                vs.append(cost(p, boxes[stem], G, G))
            para_costs.append(np.mean(vs))
            n_para += 1
    oc, pc = np.array(orig_costs), np.array(para_costs)
    out["paraphrase_robustness"] = {
        "original_mean": float(oc.mean()),
        "paraphrase_mean": float(pc.mean()),
        "n": n_para,
        "per_stimulus_abs_delta_mean": float(np.mean(np.abs(oc - pc)))}
    print(f"\n(2) paraphrase robustness (mean glimpses, n={n_para} stimuli w/ a valid reword):")
    print(f"    original expression   {oc.mean():.2f}")
    print(f"    reworded expression   {pc.mean():.2f}")
    print(f"    mean |per-stimulus change| {np.mean(np.abs(oc-pc)):.2f}")

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
