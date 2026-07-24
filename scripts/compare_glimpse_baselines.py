"""
Builds the RA-L letter's main table: glimpses-to-target for every ordering
strategy on the same stimuli, same grid, same success criterion.

Consumes:
  checkpoints/refcoco_gaze_v1.pt              our trained conditioned readout
  checkpoints/refcoco_baseline_tiles.json     SageMaker output (SUM, DeepGaze IIE)

Every model is reduced to the SAME interface -- a score per glimpse tile -- so the
ordering rule (argsort descending) is identical across models and no baseline is
handicapped by a different conversion.

  cd saliency_research && PYTHONPATH=. .venv/bin/python compare_glimpse_baselines.py
"""
import json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_TRAIN, PROC_VAL, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, DISP_W, DISP_H,
                                load_map, order_from_map, order_raster,
                                order_centre_out, order_human, glimpses, tile_of)

BASE_TILES = f"{ROOT}/checkpoints/refcoco_baseline_tiles.json"
ART_SCANPATHS = f"{ROOT}/checkpoints/art_scanpaths_val92.json"
ART_REFS = f"{ROOT}/data/refcoco_gaze/art_val_refs.json"
OUT_JSON = f"{ROOT}/checkpoints/refcoco_glimpse_table.json"
GRIDS = (4, 6)
SIGMA = 2


def order_from_tiles(scores):
    return list(np.argsort(-np.asarray(scores, float)))


def art_tile_scores():
    """ART scanpaths -> per-image tile scores.

    ART emits (x, y) in 512x320 (verified in its inference.py, not assumed). Its
    10 stochastic samples per expression are pooled into fixation counts per
    tile, the same density-style aggregation used for the human oracle, so the
    ordering rule is identical across every row of the table.
    """
    if not os.path.exists(ART_SCANPATHS):
        return {}, {}
    d = json.load(open(ART_SCANPATHS))
    W, H = d["meta"]["im_w"], d["meta"]["im_h"]
    id2img = {r["REF_ID"]: r["IMAGEFILE"] for r in json.load(open(ART_REFS))}
    tiles, hits = defaultdict(lambda: {G: np.zeros(G * G) for G in GRIDS}), defaultdict(list)
    for sp in d["scanpaths"]:
        img = id2img.get(sp["REF_ID"])
        if img is None:
            continue
        fixes = [(x, y) for wx, wy in zip(sp["X"], sp["Y"]) for x, y in zip(wx, wy)]
        for x, y in fixes:
            for G in GRIDS:
                tiles[img][G][tile_of(x / W, y / H, G)] += 1
        if fixes:                      # sanity: does ART end near the target?
            hits[img].append((fixes[-1][0] / W, fixes[-1][1] / H))
    return ({k: {G: v[G].tolist() for G in GRIDS} for k, v in tiles.items()}, hits)


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
    if not base:
        print(f"NOTE: {BASE_TILES} not found -- reporting our model + geometric "
              f"baselines only (run the SageMaker job for SUM/DeepGaze).")
    base_models = sorted({m for v in base.values() for m in v})
    art, art_last = art_tile_scores()
    if art:
        print(f"ART scanpaths loaded for {len(art)} stimuli")

    @torch.no_grad()
    def predict(entry, cond):
        im = Image.open(entry["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
        x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
        s = model.readout_forward(model.visual_features(x), cond)
        p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                        align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
        return gaussian_filter(p, SIGMA) if SIGMA else p

    rows = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(official):
        stem = e["image"].split("/")[-1]
        obs = meta.get(stem, [])
        if not obs:
            continue
        bb = obs[0]["BBOX"]
        conds = {"ours (matched)": temb[t_idx[e["intent"]]][None],
                 "ours (mismatched)": temb[t_idx[official[(i+1) % len(official)]["intent"]]][None],
                 "ours (neutral)": neutral}
        preds = {k: predict(e, c) for k, c in conds.items()}
        for G in GRIDS:
            for k, p in preds.items():
                rows[G][k].append(glimpses(order_from_map(p, G), bb, G))
            for m in base_models:
                sc = base.get(stem, {}).get(m, {}).get(str(G))
                if sc:
                    rows[G][m].append(glimpses(order_from_tiles(sc), bb, G))
            if stem in art:
                rows[G]["ART (Look Hear, ECCV'24)"].append(
                    glimpses(order_from_tiles(art[stem][G]), bb, G))
            rows[G]["raster scan"].append(glimpses(order_raster(G), bb, G))
            rows[G]["centre-out"].append(glimpses(order_centre_out(G), bb, G))
            rows[G]["human gaze (oracle)"].append(float(np.mean(
                [glimpses(order_human(obs, G, exclude=k), bb, G) for k in range(len(obs))])))

    out = {}
    for G in GRIDS:
        print(f"\n=== official-92, {G}x{G} grid ({G*G} glimpses) ===")
        print(f"{'ordering':24s} {'mean':>6s} {'median':>7s} {'succ@1':>7s} {'succ@3':>7s} {'vs raster':>10s}")
        ras = np.mean(rows[G]["raster scan"])
        order_keys = ["raster scan", "centre-out"] + base_models + \
                     ["ART (Look Hear, ECCV'24)", "ours (neutral)", "ours (mismatched)",
                      "ours (matched)", "human gaze (oracle)"]
        out[f"{G}x{G}"] = {}
        for k in order_keys:
            v = np.array(rows[G].get(k, []), float)
            if not len(v):
                continue
            out[f"{G}x{G}"][k] = {"mean": float(v.mean()), "succ@1": float((v <= 1).mean()),
                                  "succ@3": float((v <= 3).mean()), "n": len(v)}
            print(f"{k:24s} {v.mean():6.2f} {np.median(v):7.1f} "
                  f"{100*(v<=1).mean():6.0f}% {100*(v<=3).mean():6.0f}% "
                  f"{100*(1-v.mean()/ras):9.0f}%")
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
