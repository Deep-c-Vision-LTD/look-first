"""
Paper Fig. 2: glimpse orders on the 8x8 grid for three validation stimuli.

Columns: the image with the target box; our conditioned density with its first six glimpses;
ART's pooled fixations ranked by raw counts; the same fixations smoothed into a density
(sigma = 12 on a 224-pixel map). The target tile is outlined, and each panel reports its
glimpses-to-target.
"""
import json
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model_v3 import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, tile_of, order_from_map,
                                glimpses)

G = 8
# One example per regime, labelled honestly -- NOT cherry-picked all-ART-fails.
# (stem, row label): ART commits wrong / ART commits right / typical.
EXAMPLES = [("20495.jpg", ""), ("10768.jpg", ""), ("46501.jpg", "")]
ART_SIGMA = 12  # best-by-mean smoothing for ART at 8x8 (art_steelman.py)
DISP_W, DISP_H = 1680, 1050
OUT = f"{ROOT}/paper_ral/pr/fig_glimpse.png"


def main():
    import os
    os.makedirs(f"{ROOT}/paper_ral", exist_ok=True)
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
    ent = {e["image"].split("/")[-1]: e for e in official}

    raw = defaultdict(list)
    for t in json.load(open(RAW_VAL)):
        raw[t["IMAGEFILE"]].append(t)

    d = json.load(open(f"{ROOT}/checkpoints/art_scanpaths_val92.json"))
    W, H = d["meta"]["im_w"], d["meta"]["im_h"]
    id2img = {r["REF_ID"]: r["IMAGEFILE"]
              for r in json.load(open(f"{ROOT}/data/refcoco_gaze/art_val_refs.json"))}
    artfix = defaultdict(list)
    for sp in d["scanpaths"]:
        im = id2img.get(sp["REF_ID"])
        if im:
            artfix[im] += [(x / W, y / H) for wx, wy in zip(sp["X"], sp["Y"])
                           for x, y in zip(wx, wy)]

    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(12.2, 8.0))

    def draw_grid_order(ax, order, target_tile, W, H, top=6):
        # draw in DATA (pixel) coordinates so tiles align with imshow(origin='upper'):
        # image row r=0 is the TOP. (transAxes was y-flipped, putting top-row tiles at
        # the bottom of the panel.)
        tw, th = W / G, H / G
        for rank, k in enumerate(order[:top]):
            r, c = divmod(k, G)
            ax.add_patch(Rectangle((c*tw, r*th), tw, th, fill=False,
                                   edgecolor="white", lw=0.6, alpha=0.5))
            ax.text((c+0.5)*tw, (r+0.5)*th, str(rank+1),
                    color="white", fontsize=7, ha="center", va="center", weight="bold")
        r, c = divmod(target_tile, G)
        ax.add_patch(Rectangle((c*tw, r*th), tw, th, fill=False,
                               edgecolor="#00e5ff", lw=2.2))

    for row, (stem, regime) in enumerate(EXAMPLES):
        e = ent[stem]
        bb = raw[stem][0]["BBOX"]
        sent = raw[stem][0]["REF_SENTENCE"]
        img = Image.open(e["image"]).convert("RGB")
        tt = tile_of((bb[0]+bb[2]/2)/DISP_W, (bb[1]+bb[3]/2)/DISP_H, G)

        # panel 1: image + target box
        ax = axes[row, 0]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False,
                               edgecolor="#00e5ff", lw=2))
        ax.set_ylabel(f'"{sent}"', fontsize=9)
        if row == 0:
            ax.set_title("image + target", fontsize=10)

        # our density
        im2 = img.resize((SIZE, SIZE), Image.BILINEAR)
        a = (np.asarray(im2, np.float32)/255 - IM_M)/IM_S
        x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
        with torch.no_grad():
            s = model.readout_forward(model.visual_features(x), temb[t_idx[e["intent"]]][None])
            p = gaussian_filter(torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL),
                mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float64), 2)
        og = glimpses(order_from_map(p, G), bb, G)
        ax = axes[row, 1]; ax.imshow(img); ax.imshow(np.asarray(Image.fromarray(
            (p/p.max()*255).astype(np.uint8)).resize(img.size)), cmap="jet", alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        draw_grid_order(ax, order_from_map(p, G), tt, img.size[0], img.size[1])
        if row == 0:
            ax.set_title("ours (conditioned density)", fontsize=10)
        ax.set_xlabel(f"target found in {og} glimpses", fontsize=9)

        # ART fixations
        w = np.zeros(G*G)
        heat = np.zeros((G, G))
        for xn, yn in artfix[stem]:
            w[tile_of(xn, yn, G)] += 1
            heat[min(int(yn*G), G-1), min(int(xn*G), G-1)] += 1
        ag = glimpses(list(np.argsort(-w)), bb, G)
        ax = axes[row, 2]; ax.imshow(img)
        ax.imshow(np.asarray(Image.fromarray((heat/max(heat.max(), 1)*255).astype(np.uint8)
                  ).resize(img.size, Image.NEAREST)), cmap="jet", alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        draw_grid_order(ax, list(np.argsort(-w)), tt, img.size[0], img.size[1])
        if row == 0:
            ax.set_title("ART, raw fixation counts", fontsize=10)
        ax.set_xlabel(f"target found in {ag} glimpses", fontsize=9)

        # ART fixations as a smoothed density (the fair conversion)
        fm = np.zeros((EVAL, EVAL))
        for xn, yn in artfix[stem]:
            fm[min(EVAL - 1, int(yn * EVAL)), min(EVAL - 1, int(xn * EVAL))] += 1
        fd = gaussian_filter(fm, ART_SIGMA)
        dg = glimpses(order_from_map(fd, G), bb, G)
        ax = axes[row, 3]; ax.imshow(img)
        ax.imshow(np.asarray(Image.fromarray((fd/max(fd.max(), 1e-12)*255).astype(np.uint8)).resize(img.size)),
                  cmap="jet", alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        draw_grid_order(ax, order_from_map(fd, G), tt, img.size[0], img.size[1])
        if row == 0:
            ax.set_title("ART, smoothed density", fontsize=10)
        ax.set_xlabel(f"target found in {dg} glimpses", fontsize=9)
        print(f"{stem} | {sent} | ours {og} | ART raw {ag} | ART density {dg}")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
