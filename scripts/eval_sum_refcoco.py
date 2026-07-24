"""
SageMaker entry point: run the language-blind saliency baselines on RefCOCO-Gaze
stimuli and dump the per-image glimpse-tile scores the RA-L letter needs.

Why this job exists: our claim is that the REFERRING EXPRESSION is what makes the
crop ordering good. The control is a strong saliency model that cannot read the
expression at all. If SUM/DeepGaze order crops as well as we do, the letter has
no contribution.

Outputs (to SM_MODEL_DIR):
  refcoco_baseline_tiles.json   {image: {model: {"4": [16 scores], "6": [36 scores]}}}
Tile scores are summed predicted density per tile, so the glimpse ordering is
argsort(-scores) computed offline -- identical to the ordering rule used for our
own model, keeping the comparison apples to apples.

Data channel 'imgs' = the 1,891 RefCOCO-Gaze stimuli (1680x1050 jpgs).
"""
import os, glob, json, subprocess, sys, traceback
import numpy as np

IMGS = os.environ.get("SM_CHANNEL_IMGS", "./images")
OUT = os.environ.get("SM_MODEL_DIR", "./model")
GRIDS = (4, 6)
S = 384


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def tile_scores(pred):
    """summed density per tile, row-major, for each grid size"""
    n, m = pred.shape
    out = {}
    for G in GRIDS:
        w = []
        for r in range(G):
            for c in range(G):
                w.append(float(pred[int(r*n/G):int((r+1)*n/G),
                                    int(c*m/G):int((c+1)*m/G)].sum()))
        out[str(G)] = w
    return out


def run_sum(paths, res):
    sh("pip install --no-cache-dir causal-conv1d --no-build-isolation || true")
    sh("pip install --no-cache-dir mamba-ssm --no-build-isolation")
    sh("pip install --no-cache-dir git+https://github.com/Arhosseini77/SUM.git")
    import torch
    from SUM import SUM, load_and_preprocess_image, predict_saliency_map
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SUM.from_pretrained("safe-models/SUM").to(dev).eval()
    # cond 2 = e-commerce token, SUM's best on our ad benchmark; cond 1 = natural eye
    for cond, tag in ((1, "SUM_natural"), (2, "SUM_ecommerce")):
        for k, p in enumerate(paths):
            img, _ = load_and_preprocess_image(p)
            pred = np.asarray(predict_saliency_map(img, cond, model, dev), np.float64)
            res.setdefault(os.path.basename(p), {})[tag] = tile_scores(pred)
            if k % 300 == 0:
                print(f"{tag}: {k}/{len(paths)}", flush=True)
        print(f"RESULT {tag} done", flush=True)


def run_deepgaze(paths, res):
    sh("pip install --no-cache-dir git+https://github.com/matthias-k/DeepGaze.git || true")
    import torch
    from PIL import Image
    import deepgaze_pytorch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(dev).eval()
    cb = torch.zeros(1, S, S, dtype=torch.float32).to(dev)   # uniform centre bias
    with torch.no_grad():
        for k, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((S, S), Image.BILINEAR)
            x = torch.tensor(np.array(im).transpose(2, 0, 1)[None], dtype=torch.float32).to(dev)
            pred = torch.exp(model(x, cb))[0, 0].cpu().numpy().astype(np.float64)
            res.setdefault(os.path.basename(p), {})["DeepGazeIIE"] = tile_scores(pred)
            if k % 300 == 0:
                print(f"DeepGazeIIE: {k}/{len(paths)}", flush=True)
    print("RESULT DeepGazeIIE done", flush=True)


def main():
    paths = sorted(glob.glob(os.path.join(IMGS, "*.jpg")))
    print(f"stimuli: {len(paths)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for fn in (run_sum, run_deepgaze):
        try:
            fn(paths, res)
        except Exception:
            print(f"{fn.__name__} FAILED:\n{traceback.format_exc()}", flush=True)
        json.dump(res, open(os.path.join(OUT, "refcoco_baseline_tiles.json"), "w"))
    print("models per image:", sorted({m for v in res.values() for m in v}), flush=True)


if __name__ == "__main__":
    main()
