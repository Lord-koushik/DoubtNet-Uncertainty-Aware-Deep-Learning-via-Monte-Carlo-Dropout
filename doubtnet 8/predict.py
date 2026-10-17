"""Ask DoubtNet about your own image(s).

    python predict.py my_digit.png
    python predict.py photo1.jpg photo2.png --T 100 --plot
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from doubtnet.inference import doubt, verdict
from doubtnet.model import load_model, get_device
from doubtnet.preprocess import image_to_tensor
from doubtnet.data import denormalize

DEFAULT_THRESHOLDS = {"entropy": 0.5, "mutual_info": 0.05}   # used if evaluate.py hasn't run (python3 run_all.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default="checkpoints/seed0.pt")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--plot", action="store_true", help="save figures/predict_<name>.png")
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device)
    th = DEFAULT_THRESHOLDS
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            th = json.load(f)["thresholds"]

    x = torch.cat([image_to_tensor(Image.open(p), auto_invert=not args.no_invert) for p in args.images])
    results = doubt(model, x, device, args.T)
    for path, r in zip(args.images, results):
        top3 = np.argsort(r["mean_probs"])[::-1][:3]
        print(f"\n{path}")
        print(f"  prediction        : {r['pred']}  (confidence {r['confidence']:.3f})")
        print(f"  top-3             : " + ", ".join(f"{c}:{r['mean_probs'][c]:.3f}" for c in top3))
        print(f"  predictive entropy: {r['entropy']:.4f}   (threshold {th['entropy']:.4f})")
        print(f"  mutual information: {r['mutual_info']:.4f}   (threshold {th['mutual_info']:.4f})")
        print(f"  variation ratio   : {r['variation_ratio']:.3f}")
        print(f"  verdict           : {verdict(r, th)}")

    if args.plot:
        from doubtnet import plots
        os.makedirs("figures", exist_ok=True)
        names = [os.path.splitext(os.path.basename(p))[0] for p in args.images]
        out = f"figures/predict_{'_'.join(names)[:60]}.png"
        plots.sample_gallery(denormalize(x).numpy(), results, names, out, ncols=min(6, len(names)))
        print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
