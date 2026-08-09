"""
Paired significance tests on glimpses-to-target, the analysis the submitted
version of the RA-L letter lacked.

Glimpse counts are discrete, bounded, right-skewed and paired by stimulus (every
method is scored on the same 92 image-and-expression pairs), so the appropriate
test is the paired Wilcoxon signed-rank test rather than a t-test. We report:
  * the paired median difference and its bootstrap 95% CI,
  * the Wilcoxon statistic and p-value,
  * the rank-biserial correlation as an effect size,
  * Holm-corrected p-values, since several comparisons are made per grid.

Two families of comparison:
  (a) ours vs each competing ordering  -- the performance claim
  (b) matched vs mismatched expression -- the grounding claim (the paper's
      methodological point; same model, only the conditioning vector differs)

  cd saliency_research && PYTHONPATH=. .venv/bin/python significance_tests.py
"""
import json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, order_from_map, order_raster,
                                order_centre_out, order_human, glimpses, tile_of)

BASE_TILES = f"{ROOT}/checkpoints/refcoco_baseline_tiles.json"
ART_SCANPATHS = f"{ROOT}/checkpoints/art_scanpaths_val92.json"
ART_REFS = f"{ROOT}/data/refcoco_gaze/art_val_refs.json"
OUT_JSON = f"{ROOT}/checkpoints/refcoco_significance.json"
GRIDS = (4, 6, 8)
SIGMA = 2
N_BOOT = 10000


def holm(pvals):
    """Holm-Bonferroni step-down correction; returns adjusted p-values in input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def paired_report(a, b, label):
    """a = ours, b = comparison. Lower is better, so a-b < 0 favours us."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return {"label": label, "n": len(a), "note": "identical on every stimulus"}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    # rank-biserial effect size for paired Wilcoxon
    r_pos = np.sum(np.sign(nz) > 0)
    rbc = 1 - 2 * stat / (len(nz) * (len(nz) + 1) / 2) if len(nz) else 0.0
    rng = np.random.default_rng(0)
    boot = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"label": label, "n": len(a),
            "mean_ours": float(a.mean()), "mean_other": float(b.mean()),
            "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "median_diff": float(np.median(d)),
            "wilcoxon_W": float(stat), "p": float(p),
            "rank_biserial": float(rbc),
            "ours_better_frac": float(np.mean(a < b)),
            "tied_frac": float(np.mean(a == b))}


def art_tiles(G):
    d = json.load(open(ART_SCANPATHS))
    W, H = d["meta"]["im_w"], d["meta"]["im_h"]
    id2img = {r["REF_ID"]: r["IMAGEFILE"] for r in json.load(open(ART_REFS))}
    out = defaultdict(lambda: np.zeros(G * G))
    for sp in d["scanpaths"]:
        img = id2img.get(sp["REF_ID"])
        if img:
            for wx, wy in zip(sp["X"], sp["Y"]):
                for x, y in zip(wx, wy):
                    out[img][tile_of(x / W, y / H, G)] += 1
    return out


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
    neutral = temb.mean(0, keepdim=True)

    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)):
            meta[t["IMAGEFILE"]].append(t)
    base = json.load(open(BASE_TILES)) if os.path.exists(BASE_TILES) else {}
    base_models = sorted({m for v in base.values() for m in v})

    # predict once per stimulus for each conditioning, re-bin per grid
    preds = {}
    with torch.no_grad():
        for i, e in enumerate(official):
            stem = e["image"].split("/")[-1]
            if stem not in meta:
                continue
            im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            f = model.visual_features(x)
            conds = {"matched": temb[t_idx[e["intent"]]][None],
                     "mismatched": temb[t_idx[official[(i+1) % len(official)]["intent"]]][None],
                     "neutral": neutral}
            preds[stem] = {}
            for k, c in conds.items():
                s = model.readout_forward(f, c)
                p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                    align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
                preds[stem][k] = gaussian_filter(p, SIGMA) if SIGMA else p
    stems = sorted(preds)
    print(f"stimuli: {len(stems)}\n")

    out = {}
    for G in GRIDS:
        at = art_tiles(G)
        cols = defaultdict(list)
        for stem in stems:
            bb = meta[stem][0]["BBOX"]
            for k in ("matched", "mismatched", "neutral"):
                cols[f"ours_{k}"].append(glimpses(order_from_map(preds[stem][k], G), bb, G))
            cols["raster"].append(glimpses(order_raster(G), bb, G))
            cols["centre_out"].append(glimpses(order_centre_out(G), bb, G))
            cols["human"].append(float(np.mean(
                [glimpses(order_human(meta[stem], G, exclude=j), bb, G)
                 for j in range(len(meta[stem]))])))
            if stem in at:
                cols["ART"].append(glimpses(list(np.argsort(-at[stem])), bb, G))
            for m in base_models:
                sc = base.get(stem, {}).get(m, {}).get(str(G))
                if sc:
                    cols[m].append(glimpses(list(np.argsort(-np.asarray(sc, float))), bb, G))

        comparisons = [c for c in ("raster", "centre_out", "DeepGazeIIE", "SUM_ecommerce",
                                   "SUM_natural", "ART") if len(cols.get(c, [])) == len(stems)]
        reports = [paired_report(cols["ours_matched"], cols[c], f"ours vs {c}")
                   for c in comparisons]
        # grounding claim: same model, only the conditioning changes
        reports.append(paired_report(cols["ours_matched"], cols["ours_mismatched"],
                                     "ours matched vs mismatched"))
        reports.append(paired_report(cols["ours_matched"], cols["ours_neutral"],
                                     "ours matched vs neutral"))
        padj = holm([r["p"] for r in reports])
        for r, pa in zip(reports, padj):
            r["p_holm"] = float(pa)

        print(f"=== {G}x{G} grid, n={len(stems)} (lower glimpses = better) ===")
        print(f"{'comparison':32s} {'ours':>6s} {'other':>7s} {'diff':>7s} "
              f"{'95% CI':>16s} {'p (Holm)':>10s} {'eff':>6s}")
        for r in reports:
            print(f"{r['label']:32s} {r['mean_ours']:6.2f} {r['mean_other']:7.2f} "
                  f"{r['mean_diff']:+7.2f} [{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}] "
                  f"{r['p_holm']:10.2e} {r['rank_biserial']:+6.2f}")
        print()
        out[f"{G}x{G}"] = reports

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
