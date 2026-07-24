"""
Turns "fewer glimpses" into wall-clock time, which is the claim a robotics
reviewer will actually test.

Total search cost = planning cost (once per image+expression) + n_glimpses x cost
of running the downstream grounding model on one crop. This script measures the
planning cost of our readout on CPU (the stand-in for on-board compute) and on
the local accelerator, then reports total cost against ART for a range of
per-crop costs.

ART's planning cost is not measurable locally (CUDA + their env), so it is taken
from its own SageMaker inference run and stated as measured there.

  cd saliency_research && PYTHONPATH=. .venv/bin/python bench_latency.py
"""
import json, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

from src.model import IntentASPP
from train_refcoco_gaze import (ROOT, PROC_VAL, PROC_TRAIN, OUT_CKPT, SIZE, EVAL,
                                IM_M, IM_S)

N_WARM, N_RUN = 3, 20
OUT_JSON = f"{ROOT}/checkpoints/refcoco_latency.json"
# from the ART SageMaker run: 92 refs x 10 samples, autoregressive over words
ART_JOB = "refcoco-art-2026-07-24-18-53-24-922"


def bench(dev_name):
    dev = torch.device(dev_name)
    model = IntentASPP().to(dev)
    model.load_state_dict(torch.load(OUT_CKPT, map_location="cpu"), strict=False)
    model.eval()
    official = json.load(open(PROC_VAL))
    all_train = json.load(open(PROC_TRAIN))
    exprs = sorted({e["intent"] for e in all_train} | {e["intent"] for e in official})
    with torch.no_grad():
        temb = model.text_embed(exprs[:64], dev)
    cond = temb[:1]

    ims = []
    for e in official[:N_RUN + N_WARM]:
        im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
        ims.append(torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev))

    def once(x):
        with torch.no_grad():
            f = model.visual_features(x)
            s = model.readout_forward(f, cond)
            return torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                               align_corners=False))

    for x in ims[:N_WARM]:
        once(x)
    if dev_name == "mps":
        torch.mps.synchronize()

    # split backbone vs readout: the backbone runs once per image, the readout
    # once per expression, so a robot re-querying the same frame pays only the latter
    t0 = time.perf_counter()
    for x in ims[N_WARM:]:
        with torch.no_grad():
            model.visual_features(x)
    if dev_name == "mps":
        torch.mps.synchronize()
    t_backbone = (time.perf_counter() - t0) / N_RUN

    with torch.no_grad():
        feats = [model.visual_features(x) for x in ims[N_WARM:]]
    if dev_name == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for f in feats:
        with torch.no_grad():
            s = model.readout_forward(f, cond)
            torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                        align_corners=False))
    if dev_name == "mps":
        torch.mps.synchronize()
    t_readout = (time.perf_counter() - t0) / N_RUN
    return t_backbone, t_readout


def main():
    res = {}
    for d in ("cpu",) + (("mps",) if torch.backends.mps.is_available() else ()):
        tb, tr = bench(d)
        res[d] = {"backbone_s": tb, "readout_s": tr, "total_s": tb + tr}
        print(f"{d:4s}: backbone {1000*tb:7.1f} ms | readout {1000*tr:6.2f} ms "
              f"| total {1000*(tb+tr):7.1f} ms per (image, expression)", flush=True)
    print("\nRe-querying the SAME frame with a new expression costs only the readout: "
          f"{1000*res[list(res)[-1]]['readout_s']:.2f} ms", flush=True)

    # total search cost vs ART at 8x8, using the measured glimpse counts
    dist = json.load(open(f"{ROOT}/checkpoints/refcoco_glimpse_distributions.json"))
    ours_n, art_n = dist["8x8"]["ours"]["mean"], dist["8x8"]["ART"]["mean"]
    plan = res["cpu"]["total_s"]
    print(f"\ntotal search cost at 8x8 (planning + n x per-crop grounding), "
          f"ours n={ours_n:.1f} vs ART n={art_n:.1f}:")
    print(f"{'per-crop cost':>14s} {'ours':>10s} {'ART':>10s} {'saving':>8s}")
    for t_crop in (0.010, 0.030, 0.100, 0.300):
        o = plan + ours_n * t_crop
        a = art_n * t_crop          # ART's own planning cost excluded (favours ART)
        print(f"{1000*t_crop:11.0f} ms {o:9.2f}s {a:9.2f}s {100*(1-o/a):7.0f}%")
    res["search_cost_note"] = ("ART planning cost excluded from its total, which "
                               "favours ART; ours includes full planning")
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
