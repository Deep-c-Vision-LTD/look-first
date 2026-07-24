"""
Is ART's heavy tail a property of the model, or an artefact of one grid choice?

Recomputes the full distribution of glimpses-to-target at 4x4, 6x6 and 8x8 for
our conditioned readout, ART, the human oracle and raster scan. ART's raw
fixations and our predicted maps are both re-binned at each grid, so nothing is
locked to the grid used in the main table.

(SUM / DeepGaze are absent here: the SageMaker job emitted tile scores only for
the 4x4 and 6x6 grids, and re-running them for 8x8 would not change what this
analysis is asking.)

  cd saliency_research && PYTHONPATH=. .venv/bin/python analyze_glimpse_distributions.py
"""
import json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_TRAIN, PROC_VAL, RAW_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, load_map, order_from_map,
                                order_raster, order_human, glimpses, tile_of)

ART_SCANPATHS = f"{ROOT}/checkpoints/art_scanpaths_val92.json"
ART_REFS = f"{ROOT}/data/refcoco_gaze/art_val_refs.json"
OUT_JSON = f"{ROOT}/checkpoints/refcoco_glimpse_distributions.json"
GRIDS = (4, 6, 8)
SIGMA = 2


def art_fixations():
    d = json.load(open(ART_SCANPATHS))
    W, H = d["meta"]["im_w"], d["meta"]["im_h"]
    id2img = {r["REF_ID"]: r["IMAGEFILE"] for r in json.load(open(ART_REFS))}
    out = defaultdict(list)
    for sp in d["scanpaths"]:
        img = id2img.get(sp["REF_ID"])
        if img:
            out[img] += [(x / W, y / H) for wx, wy in zip(sp["X"], sp["Y"])
                         for x, y in zip(wx, wy)]
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

    meta = defaultdict(list)
    for f in (RAW_TRAIN, RAW_VAL):
        for t in json.load(open(f)):
            meta[t["IMAGEFILE"]].append(t)
    artfix = art_fixations()

    # predict once per stimulus, re-bin per grid
    preds, boxes, obss = {}, {}, {}
    with torch.no_grad():
        for e in official:
            stem = e["image"].split("/")[-1]
            if stem not in meta or stem not in artfix:
                continue
            im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            s = model.readout_forward(model.visual_features(x), temb[t_idx[e["intent"]]][None])
            p = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                            align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
            preds[stem] = gaussian_filter(p, SIGMA) if SIGMA else p
            boxes[stem] = meta[stem][0]["BBOX"]
            obss[stem] = meta[stem]
    print(f"stimuli with both our prediction and ART scanpaths: {len(preds)}\n")

    out = {}
    for G in GRIDS:
        res = defaultdict(list)
        for stem, p in preds.items():
            bb = boxes[stem]
            res["ours"].append(glimpses(order_from_map(p, G), bb, G))
            w = np.zeros(G * G)
            for xn, yn in artfix[stem]:
                w[tile_of(xn, yn, G)] += 1
            res["ART"].append(glimpses(list(np.argsort(-w)), bb, G))
            res["raster"].append(glimpses(order_raster(G), bb, G))
            res["human"].append(float(np.mean(
                [glimpses(order_human(obss[stem], G, exclude=k), bb, G)
                 for k in range(len(obss[stem]))])))
        half = G * G / 2
        print(f"=== {G}x{G} ({G*G} tiles), n={len(res['ours'])} ===")
        print(f"{'':7s} {'mean':>6s} {'median':>7s} {'p75':>6s} {'p90':>6s} {'max':>5s} {'>half grid':>11s}")
        out[f"{G}x{G}"] = {}
        for k in ("raster", "ART", "ours", "human"):
            v = np.array(res[k], float)
            out[f"{G}x{G}"][k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                  "p90": float(np.percentile(v, 90)),
                                  "frac_over_half": float(np.mean(v > half))}
            print(f"{k:7s} {v.mean():6.2f} {np.median(v):7.1f} {np.percentile(v,75):6.1f} "
                  f"{np.percentile(v,90):6.1f} {v.max():5.0f} {100*np.mean(v>half):10.0f}%")
        o, a = np.array(res["ours"], float), np.array(res["ART"], float)
        out[f"{G}x{G}"]["paired"] = {"ours_better": float(np.mean(o < a)),
                                     "tied": float(np.mean(o == a)),
                                     "art_better": float(np.mean(o > a))}
        print(f"        paired vs ART: ours better {100*np.mean(o<a):.0f}%, "
              f"tied {100*np.mean(o==a):.0f}%, ART better {100*np.mean(o>a):.0f}%\n")

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
