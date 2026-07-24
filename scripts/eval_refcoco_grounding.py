"""
Decisive control for the RA-L glimpse-planning claim: does the referring
expression actually drive WHERE the model looks first, or does it just produce a
generically good saliency map?

Compares, on the same stimuli and the same trained readout:
  matched      - the stimulus's own referring expression
  mismatched   - another stimulus's expression (paired, deterministic shift)
  neutral      - mean text embedding (a valid in-distribution "no specific goal"
                 conditioning vector, unlike a zero vector fed to a trained FiLM)

Reports CC and glimpses-to-target side by side, plus the paired win-rate, which
is what the letter's claim actually rests on.

  cd saliency_research && PYTHONPATH=. .venv/bin/python eval_refcoco_grounding.py
"""
import json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_TRAIN, PROC_VAL, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, N_INTERNAL_VAL, DISP_W, DISP_H,
                                cc_np, load_map, order_from_map, order_raster,
                                order_centre_out, order_human, glimpses)

GRIDS = (4, 6)
SIGMA = 2


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = IntentASPP().to(dev)
    sd = torch.load(OUT_CKPT, map_location="cpu")
    missing = model.load_state_dict(sd, strict=False)
    print(f"loaded readout from {OUT_CKPT}", flush=True)
    model.eval()

    all_train = json.load(open(PROC_TRAIN))
    official = json.load(open(PROC_VAL))
    perm = np.random.default_rng(0).permutation(len(all_train))
    internal = [all_train[i] for i in perm[:N_INTERNAL_VAL]]

    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)):
            meta[t["IMAGEFILE"]].append(t)

    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official})
    with torch.no_grad():
        embs = [model.text_embed(exprs[i:i + 256], dev).cpu() for i in range(0, len(exprs), 256)]
    temb = torch.cat(embs).to(dev)
    t_idx = {s: i for i, s in enumerate(exprs)}
    neutral = temb.mean(0, keepdim=True)      # in-distribution "no specific goal"

    @torch.no_grad()
    def predict(entry, cond):
        im = Image.open(entry["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
        x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
        f = model.visual_features(x)
        s = model.readout_forward(f, cond)
        p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                        align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
        return gaussian_filter(p, SIGMA) if SIGMA else p

    def run(entries, tag):
        rows = defaultdict(list)
        for i, e in enumerate(entries):
            stem = e["image"].split("/")[-1]
            obs = meta.get(stem, [])
            if not obs:
                continue
            bb = obs[0]["BBOX"]
            gt = load_map(e["gt"], EVAL)
            conds = {
                "matched": temb[t_idx[e["intent"]]][None],
                "mismatched": temb[t_idx[entries[(i + 1) % len(entries)]["intent"]]][None],
                "neutral": neutral,
            }
            for name, c in conds.items():
                p = predict(e, c)
                rows[f"cc_{name}"].append(cc_np(p, gt))
                for G in GRIDS:
                    rows[f"g{G}_{name}"].append(glimpses(order_from_map(p, G), bb, G))
            for G in GRIDS:
                rows[f"g{G}_raster"].append(glimpses(order_raster(G), bb, G))
                rows[f"g{G}_centre"].append(glimpses(order_centre_out(G), bb, G))
                rows[f"g{G}_human"].append(float(np.mean(
                    [glimpses(order_human(obs, G, exclude=k), bb, G) for k in range(len(obs))])))

        n = len(rows["cc_matched"])
        print(f"\n=== {tag} (n={n}) ===")
        print(f"CC        matched {np.mean(rows['cc_matched']):.4f} | "
              f"mismatched {np.mean(rows['cc_mismatched']):.4f} | "
              f"neutral {np.mean(rows['cc_neutral']):.4f}")
        for G in GRIDS:
            m = np.array(rows[f"g{G}_matched"], float)
            mm = np.array(rows[f"g{G}_mismatched"], float)
            nu = np.array(rows[f"g{G}_neutral"], float)
            ra = np.array(rows[f"g{G}_raster"], float)
            hu = np.array(rows[f"g{G}_human"], float)
            win = float(np.mean(m < mm)); tie = float(np.mean(m == mm))
            print(f"glimpses {G}x{G}: matched {m.mean():.2f} | mismatched {mm.mean():.2f} | "
                  f"neutral {nu.mean():.2f} | raster {ra.mean():.2f} | human {hu.mean():.2f}")
            print(f"   paired: matched better on {100*win:.0f}% of stimuli, tied {100*tie:.0f}%, "
                  f"worse {100*(1-win-tie):.0f}%  | succ@1 matched {100*np.mean(m<=1):.0f}% "
                  f"vs mismatched {100*np.mean(mm<=1):.0f}%")
        return {k: float(np.mean(v)) for k, v in rows.items()}

    out = {"internal_299": run(internal, "internal-299"),
           "official_92": run(official, "official-92")}
    json.dump(out, open(f"{ROOT}/checkpoints/refcoco_grounding_controls.json", "w"), indent=2)
    print("\nsaved -> checkpoints/refcoco_grounding_controls.json")


if __name__ == "__main__":
    main()
