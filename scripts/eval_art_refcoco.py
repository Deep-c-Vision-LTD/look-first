"""
SageMaker entry point: run ART ("Look Hear", ECCV 2024, the RefCOCO-Gaze authors'
own model) on our 92 official-val referring expressions and dump its predicted
scanpaths, so they can be converted offline into glimpse orderings and compared
with our conditioned readout under an identical protocol.

This is the MANDATORY baseline for the RA-L letter: omitting the dataset owners'
own model is exactly the omission that caused a desk rejection in June 2026.

Notes grounded in ART's actual source (not guessed):
  * inference.py emits [min(im_w-1,int(x)), min(im_h-1,int(y))] -> (x, y) with x
    clamped by WIDTH, in 512x320 space. No transpose.
  * main() only needs REF_ID / IMAGEFILE / REF_WORDS per ref, so their
    preprocessed teacher-forcing json is not required.
  * get_metrics() is skipped: it scores against their withheld test scanpaths,
    which we neither have nor need (we score glimpse orderings ourselves).
  * env is python3.8 / torch 1.12 / cu113 to match art_env_export.yml.

Channels: 'imgs' = 512x320 stimuli, 'refs' = art_val_refs.json
"""
import os, sys, json, subprocess, traceback, shutil

REFS = os.environ.get("SM_CHANNEL_REFS", "./refs")
IMGS = os.environ.get("SM_CHANNEL_IMGS", "./images")
OUT = os.environ.get("SM_MODEL_DIR", "./model")
REPO = "/opt/ml/ART"

DRIVE = {  # file ids resolved from the authors' public Drive folder
    "checkpoints/art_checkpoint.pkg": "1ZIhNOoa3Jn0XjXkNAzvNOQJ25MhAqJOS",
    "data/catDict.pkl": "1v1Kh2louU7tZbTJMqd-WpcsbXZvNvvb0",
    "data/clusters_refcocogaze.npy": "1JrPJ8F6i6CbzhKRoWkfIY3bGIZigyO6P",
}
NUM_SAMPLES = 10   # ART's own default: 10 stochastic scanpaths per referring expression


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=check)


def setup():
    sh("pip install --no-cache-dir gdown 'transformers==4.30.2' tqdm 'numpy<1.24'")
    sh(f"git clone --depth 1 https://github.com/cvlab-stonybrook/ART.git {REPO}")
    for rel, fid in DRIVE.items():
        dst = os.path.join(REPO, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sh(f"gdown --id {fid} -O {dst}")
        print(f"  {rel}: {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
    # our stimuli, already resized to ART's 512x320 (same 1.6 aspect, no padding)
    dst = os.path.join(REPO, "data", "images_512X320")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in os.listdir(IMGS):
        if f.endswith(".jpg"):
            shutil.copy(os.path.join(IMGS, f), os.path.join(dst, f))
            n += 1
    print(f"staged {n} stimuli into {dst}", flush=True)


def main():
    setup()
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import torch
    from transformers import AutoTokenizer
    from utils.core_utils import seed_everything, get_args_parser_test
    from refgaze import RefGaze
    import inference as art_inference

    seed_everything(42)
    parser = get_args_parser_test()
    args, _ = parser.parse_known_args([])
    args.img_dir = os.path.join(REPO, "data", "images_512X320")
    args.cat_dict_file = os.path.join(REPO, "data", "catDict.pkl")
    args.dataset_dir = os.path.join(REPO, "data")
    args.num_samples = NUM_SAMPLES
    print(f"args: im_w={args.im_w} im_h={args.im_h} lm={args.lm} "
          f"num_samples={args.num_samples}", flush=True)

    refs = json.load(open(os.path.join(REFS, "art_val_refs.json")))
    test_refs = [{"REF_ID": r["REF_ID"], "IMAGEFILE": r["IMAGEFILE"],
                  "REF_WORDS": r["REF_WORDS"]} for r in refs]
    print(f"refs to run: {len(test_refs)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    model = RefGaze(args=args).cuda()
    model.eval()
    ckpt = torch.load(os.path.join(REPO, "checkpoints", "art_checkpoint.pkg"),
                      map_location="cpu")
    model.load_state_dict(ckpt["model"])
    del ckpt
    print("ART checkpoint loaded", flush=True)

    results = art_inference.generate_scanpaths(model=model, tokenizer=tokenizer,
                                               test_refs=test_refs,
                                               num_samples=args.num_samples, args=args)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"scanpaths": results,
               "meta": {"im_w": args.im_w, "im_h": args.im_h,
                        "num_samples": args.num_samples,
                        "coord_order": "x,y (x clamped by width) in 512x320"}},
              open(os.path.join(OUT, "art_scanpaths_val92.json"), "w"))
    nfix = sum(len([f for w in r["X"] for f in w]) for r in results)
    print(f"RESULT ART: {len(results)} scanpaths ({len(test_refs)} refs x "
          f"{NUM_SAMPLES} samples), {nfix} fixations total", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ART EVAL FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
