"""
Extends the evaluation from the 92-pair official validation split to 391 pairs,
addressing the most obvious objection to the submitted version (a headline
resting on 92 stimuli).

Which comparisons can legitimately be extended, and which cannot:
  * ours          -- the internal-299 split is held out from OUR training
                     (1500 train / 299 internal-val, seed 0), so it is fair.
  * SUM, DeepGaze -- zero-shot, never trained on RefCOCO-Gaze, so fair anywhere.
                     The SageMaker job scored all 1,891 stimuli, so the tiles exist.
  * human oracle  -- leave-one-observer-out, fair anywhere.
  * ART           -- CANNOT be extended. ART was trained on the RefCOCO-Gaze train
                     split, from which internal-299 is carved. The ART comparison
                     therefore stays on the official 92 only, and the paper must
                     say so.

  cd saliency_research && PYTHONPATH=. .venv/bin/python eval_extended_split.py
"""
import json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, N_INTERNAL_VAL, order_from_map,
                                order_raster, order_centre_out, order_human, glimpses)

BASE_TILES = f"{ROOT}/checkpoints/refcoco_baseline_tiles.json"
OUT_JSON = f"{ROOT}/checkpoints/refcoco_extended_split.json"
GRIDS = (4, 6)          # SUM/DeepGaze tiles exist for these grids
SIGMA = 2


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = IntentASPP().to(dev)
    model.load_state_dict(torch.load(OUT_CKPT, map_location="cpu"), strict=False)
    model.eval()

    all_train = json.load(open(PROC_TRAIN))
    official = json.load(open(PROC_VAL))
    perm = np.random.default_rng(0).permutation(len(all_train))
    internal = [all_train[i] for i in perm[:N_INTERNAL_VAL]]
    print(f"internal-299: {len(internal)}   official-92: {len(official)}")

    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official})
    with torch.no_grad():
        temb = torch.cat([model.text_embed(exprs[i:i+256], dev).cpu()
                          for i in range(0, len(exprs), 256)]).to(dev)
    t_idx = {s: i for i, s in enumerate(exprs)}
    neutral = temb.mean(0, keepdim=True)

    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)):
            meta[t["IMAGEFILE"]].append(t)
    base = json.load(open(BASE_TILES))
    base_models = sorted({m for v in base.values() for m in v})

    def run(entries, tag):
        cols = defaultdict(list)
        with torch.no_grad():
            for i, e in enumerate(entries):
                stem = e["image"].split("/")[-1]
                if stem not in meta or stem not in base:
                    continue
                bb = meta[stem][0]["BBOX"]
                im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
                x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
                f = model.visual_features(x)
                conds = {"matched": temb[t_idx[e["intent"]]][None],
                         "mismatched": temb[t_idx[entries[(i+1) % len(entries)]["intent"]]][None],
                         "neutral": neutral}
                pr = {}
                for k, c in conds.items():
                    s = model.readout_forward(f, c)
                    p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                        align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
                    pr[k] = gaussian_filter(p, SIGMA) if SIGMA else p
                for G in GRIDS:
                    for k in conds:
                        cols[(G, f"ours_{k}")].append(glimpses(order_from_map(pr[k], G), bb, G))
                    cols[(G, "raster")].append(glimpses(order_raster(G), bb, G))
                    cols[(G, "centre_out")].append(glimpses(order_centre_out(G), bb, G))
                    cols[(G, "human")].append(float(np.mean(
                        [glimpses(order_human(meta[stem], G, exclude=j), bb, G)
                         for j in range(len(meta[stem]))])))
                    for m in base_models:
                        sc = base[stem][m][str(G)]
                        cols[(G, m)].append(glimpses(list(np.argsort(-np.asarray(sc, float))), bb, G))
        return cols

    ci = run(internal, "internal-299")
    co = run(official, "official-92")

    out = {}
    for G in GRIDS:
        n_i = len(ci[(G, "ours_matched")]); n_o = len(co[(G, "ours_matched")])
        print(f"\n=== {G}x{G} grid ===")
        print(f"{'ordering':22s} {'internal-299':>13s} {'official-92':>12s} {'combined-391':>13s}")
        out[f"{G}x{G}"] = {}
        keys = ["raster", "centre_out"] + base_models + \
               ["ours_neutral", "ours_mismatched", "ours_matched", "human"]
        for k in keys:
            a = np.array(ci[(G, k)], float); b = np.array(co[(G, k)], float)
            comb = np.concatenate([a, b])
            out[f"{G}x{G}"][k] = {"internal_299": float(a.mean()), "official_92": float(b.mean()),
                                  "combined_391": float(comb.mean()), "n": len(comb)}
            print(f"{k:22s} {a.mean():13.2f} {b.mean():12.2f} {comb.mean():13.2f}")
        # significance on the combined split, ours vs each competitor
        m = np.concatenate([ci[(G, "ours_matched")], co[(G, "ours_matched")]])
        print(f"\n  paired Wilcoxon on combined n={len(m)} (ours matched vs):")
        for k in ["raster", "centre_out"] + base_models + ["ours_mismatched", "ours_neutral"]:
            o = np.concatenate([ci[(G, k)], co[(G, k)]])
            _, p = wilcoxon(m, o)
            out[f"{G}x{G}"][k]["p_vs_ours_combined"] = float(p)
            print(f"    {k:22s} diff {m.mean()-o.mean():+7.2f}   p = {p:.2e}")
    print(f"\nNOTE: ART is absent by design; it trained on the RefCOCO-Gaze train split "
          f"from which internal-299 is drawn, so it can only be compared on official-92.")
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
