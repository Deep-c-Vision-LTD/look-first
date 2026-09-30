"""
Scanpath-to-order conversion analysis for ART (paper Table 3).

ART emits fixation points, not a density, so scoring it by search cost needs a rule that turns
fixations into a ranking of tiles. This script scores ART's fixations under raw-count rankings
(three tie-breaking rules) and under Gaussian-smoothed densities (bandwidth swept), and tests
our readout against each variant with a paired Wilcoxon test. The bandwidth reported in the
paper is chosen on the evaluation stimuli themselves, which favours ART.

  cd saliency_research && PYTHONPATH=. .venv/bin/python scripts/art_steelman.py
"""
import json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from src.model_v3 import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_TRAIN, PROC_VAL, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, order_from_map, order_raster,
                                glimpses, tile_of)
from analyze_glimpse_distributions import art_fixations, SIGMA

GRIDS = (4, 6, 8)
ART_SIGMAS = (2, 4, 8, 12, 16, 24, 32)

def stats(v, G):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "frac_over_half": float(np.mean(v > G * G / 2))}

def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = IntentASPP().to(dev)
    model.load_state_dict(torch.load(OUT_CKPT, map_location="cpu"), strict=False); model.eval()
    official = json.load(open(PROC_VAL)); all_train = json.load(open(PROC_TRAIN))
    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official})
    with torch.no_grad():
        temb = torch.cat([model.text_embed(exprs[i:i+256], dev).cpu() for i in range(0, len(exprs), 256)]).to(dev)
    t_idx = {s: i for i, s in enumerate(exprs)}
    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)): meta[t["IMAGEFILE"]].append(t)
    artfix = art_fixations()
    preds, boxes = {}, {}
    with torch.no_grad():
        for e in official:
            stem = e["image"].split("/")[-1]
            if stem not in meta or stem not in artfix: continue
            im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            s = model.readout_forward(model.visual_features(x), temb[t_idx[e["intent"]]][None])
            p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
            preds[stem] = gaussian_filter(p, SIGMA) if SIGMA else p
            boxes[stem] = meta[stem][0]["BBOX"]
    stems = sorted(preds)
    print(f"n = {len(stems)}")
    out = {"n": len(stems), "art_sigmas": list(ART_SIGMAS)}
    for G in GRIDS:
        cen = np.array([((c + .5) / G - .5) ** 2 + ((r + .5) / G - .5) ** 2 for r in range(G) for c in range(G)])
        ours, var = [], defaultdict(list)
        for st in stems:
            bb = boxes[st]
            ours.append(glimpses(order_from_map(preds[st], G), bb, G))
            w = np.zeros(G * G)
            for xn, yn in artfix[st]: w[tile_of(xn, yn, G)] += 1
            var["raw, default sort order"].append(glimpses(list(np.argsort(-w)), bb, G))
            var["raw, stable index tie-break"].append(glimpses(list(np.argsort(-w, kind="stable")), bb, G))
            var["raw, centre-out tie-break"].append(glimpses(list(np.lexsort((cen, -w))), bb, G))
            fm = np.zeros((EVAL, EVAL))
            for xn, yn in artfix[st]:
                fm[min(EVAL - 1, int(yn * EVAL)), min(EVAL - 1, int(xn * EVAL))] += 1
            for sg in ART_SIGMAS:
                var[f"density sigma={sg}"].append(glimpses(order_from_map(gaussian_filter(fm, sg), G), bb, G))
        o = np.asarray(ours, float)
        rows = {}
        for k, v in var.items():
            a = np.asarray(v, float); d = o - a
            p = float(wilcoxon(o, a).pvalue) if np.any(d != 0) else 1.0
            rows[k] = {**stats(a, G), "ours_better": float(np.mean(o < a)), "tied": float(np.mean(o == a)),
                       "art_better": float(np.mean(o > a)), "p_wilcoxon_ours_vs_art": p}
        best = min((k for k in rows), key=lambda k: rows[k]["mean"])
        out[f"{G}x{G}"] = {"ours": stats(o, G), "art_variants": rows, "art_best_by_mean": best}
        print(f"\n=== {G}x{G} === ours: mean {o.mean():.2f} med {np.median(o):.1f} p90 {np.percentile(o,90):.1f} >half {100*np.mean(o>G*G/2):.0f}%")
        for k, r in rows.items():
            flag = "  <- ART best" if k == best else ""
            print(f"  ART {k:30s} mean {r['mean']:6.2f} med {r['median']:5.1f} p90 {r['p90']:5.1f} >half {100*r['frac_over_half']:3.0f}% | ours better {100*r['ours_better']:3.0f}% ART better {100*r['art_better']:3.0f}% p={r['p_wilcoxon_ours_vs_art']:.3g}{flag}")
    json.dump(out, open(f"{ROOT}/checkpoints/refcoco_art_steelman.json", "w"), indent=1)
    print("\nwrote checkpoints/refcoco_art_steelman.json")

if __name__ == "__main__":
    main()
