"""
RA-L Figure 1: why a hedged density beats a committed scanpath at fine granularity.

Two stimuli where the referring expression carries a spatial cue ("... on right").
For each: the image with the target box; our predicted density with the 8x8
glimpse order numbered on the tiles it visits first; ART's pooled fixations with
its glimpse order. The target tile is outlined. The caption numbers (ours vs ART
glimpses-to-target) come straight from the results json.
"""
import json
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, RAW_VAL, OUT_CKPT,
                                SIZE, EVAL, IM_M, IM_S, tile_of, order_from_map,
                                glimpses)

G = 8
# One example per regime, labelled honestly -- NOT cherry-picked all-ART-fails.
# (stem, row label): ART commits wrong / ART commits right / typical.
EXAMPLES = [("20495.jpg", "ART misgrounds: catastrophic"),
            ("10768.jpg", "ART grounds correctly: near-instant"),
            ("46501.jpg", "ours hedges: consistently low")]
DISP_W, DISP_H = 1680, 1050
OUT = f"{ROOT}/paper_ral/fig_glimpse.png"


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

    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(9.2, 8.0))

    def draw_grid_order(ax, order, target_tile, top=6):
        for rank, k in enumerate(order[:top]):
            r, c = divmod(k, G)
            ax.add_patch(Rectangle((c/G, r/G), 1/G, 1/G, transform=ax.transAxes,
                                   fill=False, edgecolor="white", lw=0.6, alpha=0.5))
            ax.text((c+0.5)/G, (r+0.5)/G, str(rank+1), transform=ax.transAxes,
                    color="white", fontsize=7, ha="center", va="center", weight="bold")
        r, c = divmod(target_tile, G)
        ax.add_patch(Rectangle((c/G, r/G), 1/G, 1/G, transform=ax.transAxes,
                               fill=False, edgecolor="#00e5ff", lw=2.2))

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
        ax.set_ylabel(f'"{sent}"\n[{regime}]', fontsize=8)
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
        draw_grid_order(ax, order_from_map(p, G), tt)
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
        draw_grid_order(ax, list(np.argsort(-w)), tt)
        if row == 0:
            ax.set_title("ART (Look Hear) fixations", fontsize=10)
        ax.set_xlabel(f"target found in {ag} glimpses", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
