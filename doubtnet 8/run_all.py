"""One command for the whole study.

    python3 check_env.py            # run first - diagnoses problems
    python3 run_all.py              # full: 3 networks x 6 epochs, T=30   (~15-25 min on a Mac)
    python3 run_all.py --fast       # 3 networks x 3 epochs, T=20     (~5-8 min)  -> real results, quicker
    python3 run_all.py --quick      # 2 networks x 1 epoch,  T=10         (~3-5 min)  -> test the pipeline

Re-running skips networks that are already trained, so it is safe to stop (Ctrl+C) and resume.
At the end it prints a summary and writes results_pack.zip (report + figures) ready to share.
"""
import argparse
import json
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="number of networks (also the ensemble size)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--T", type=int, default=30, help="MC Dropout passes")
    ap.add_argument("--n-ood", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="data")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="~5-8 min: 3 networks x 3 epochs, T=20, 1000 OOD images, 100 bootstraps")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.seeds, a.epochs, a.T, a.n_ood, a.boot = 3, 3, 20, 1000, 100
    if a.quick:
        a.seeds, a.epochs, a.T, a.n_ood, a.boot = 2, 1, 10, 300, 30

    t0 = time.time()
    from doubtnet.training import train_model
    from doubtnet import experiments, plots

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    if not a.skip_train:
        print(f"== Training {a.seeds} networks ({a.epochs} epochs each) ==", flush=True)
        for s in range(a.seeds):
            out = f"checkpoints/seed{s}.pt"
            if os.path.exists(out):
                print(f"  seed {s}: finished checkpoint exists, skipping", flush=True)
                continue
            hist = train_model(s, a.epochs, out, a.data, device=a.device)
            if s == 0:
                plots.training_curves(hist, "figures/training_curves.png")
                with open("checkpoints/history_seed0.json", "w") as f:
                    json.dump(hist, f)
    print("\n== Experiments ==", flush=True)
    R = experiments.run("checkpoints", a.T, a.n_ood, a.data, a.device, a.boot)
    experiments.print_summary(R)

    shutil.make_archive("results_pack", "zip", root_dir=".", base_dir="results")
    with __import__("zipfile").ZipFile("results_pack.zip", "a") as z:
        for fn in sorted(os.listdir("figures")):
            z.write(os.path.join("figures", fn))
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print("Send me:  results_pack.zip   (contains results/ and figures/)")


if __name__ == "__main__":
    main()
