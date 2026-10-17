"""Train ONE DoubtNet (for quick tests). For the full study use:  python run_all.py"""
import argparse

from doubtnet.training import train_model
from doubtnet import plots
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-conv", type=float, default=0.25)
    ap.add_argument("--p-fc", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data")
    ap.add_argument("--quick", action="store_true", help="1 epoch smoke test")
    a = ap.parse_args()
    out = f"checkpoints/seed{a.seed}.pt"
    hist = train_model(a.seed, 1 if a.quick else a.epochs, out, a.data, p_conv=a.p_conv,
                       p_fc=a.p_fc, device=a.device)
    os.makedirs("figures", exist_ok=True)
    plots.training_curves(hist, "figures/training_curves.png")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
