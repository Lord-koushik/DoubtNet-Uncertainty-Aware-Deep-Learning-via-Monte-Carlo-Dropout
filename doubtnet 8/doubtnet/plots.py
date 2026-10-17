"""All figures for the report. Uses the non-interactive Agg backend."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def training_curves(history, path):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, history["train_loss"], color=BLUE, label="train")
    ax[0].plot(ep, history["val_loss"], color=ORANGE, label="validation")
    ax[0].set(title="Cross-entropy loss", xlabel="epoch")
    ax[1].plot(ep, np.array(history["train_acc"]) * 100, color=BLUE, label="train")
    ax[1].plot(ep, np.array(history["val_acc"]) * 100, color=ORANGE, label="validation")
    ax[1].set(title="Accuracy (%)", xlabel="epoch")
    for a in ax:
        a.legend()
    _save(fig, path)


def reliability_diagram(det_probs, mc_probs_mean, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, probs, name, c in [(axes[0], det_probs, "Deterministic (dropout off)", GREY),
                               (axes[1], mc_probs_mean, "MC Dropout", BLUE)]:
        e, centers, acc, conf, cnt = metrics.reliability(probs, labels)
        m = cnt > 0
        ax.bar(centers[m], acc[m], width=1 / len(centers), color=c, alpha=0.8,
               edgecolor="white", label="accuracy")
        ax.plot([0, 1], [0, 1], "--", color="black", lw=1, label="perfect calibration")
        ax.set(title=f"{name}\nECE = {e * 100:.2f}%", xlabel="confidence", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("accuracy")
    _save(fig, path)


def uncertainty_histograms(in_unc, ood_unc: dict, score_name, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(np.max(in_unc), *[np.max(v) for v in ood_unc.values()]) + 1e-6, 50)
    ax.hist(in_unc, bins=bins, density=True, alpha=0.55, color=BLUE, label="MNIST (in-distribution)")
    cmap = plt.get_cmap("tab10")
    for i, (name, u) in enumerate(ood_unc.items()):
        ax.hist(u, bins=bins, density=True, histtype="step", lw=1.6, color=cmap(i + 1), label=name)
    ax.set(title=f"{score_name}: MNIST vs out-of-distribution", xlabel=score_name, ylabel="density")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    _save(fig, path)


def ood_auroc_bars(table, path):
    """table: {dataset: {score: auroc}}"""
    names = list(table)
    scores = list(next(iter(table.values())))
    x = np.arange(len(names))
    w = 0.8 / len(scores)
    colors = [GREY, BLUE, ORANGE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, s in enumerate(scores):
        ax.bar(x + i * w - 0.4 + w / 2, [table[n][s] * 100 for n in names], w,
               label=s, color=colors[i % len(colors)])
    ax.axhline(50, color="black", lw=0.8, ls=":")
    ax.set_xticks(x, names, rotation=15)
    ax.set(ylabel="AUROC (%)", ylim=(40, 101), title="How well does each score detect OOD inputs?")
    ax.legend(fontsize=8, ncol=len(scores), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)


def rotation_curve(angles, acc, ent, mi, path):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(angles, np.array(acc) * 100, "o-", color=GREY, label="accuracy (%)")
    ax1.set(xlabel="rotation angle (degrees)", ylabel="accuracy (%)", ylim=(0, 100))
    ax2 = ax1.twinx()
    ax2.plot(angles, ent, "s-", color=BLUE, label="predictive entropy")
    ax2.plot(angles, mi, "^-", color=ORANGE, label="mutual information")
    ax2.set_ylabel("mean uncertainty (nats)")
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("Dataset shift: as digits rotate, accuracy falls and doubt rises")
    _save(fig, path)


def rejection_plot(curves: dict, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [GREY, BLUE, ORANGE, GREEN]
    for (name, (f, a)), c in zip(curves.items(), colors):
        ax.plot(f * 100, a * 100, "o-", ms=3, color=c, label=name)
    ax.set(xlabel="% of test inputs refused (most uncertain first)",
           ylabel="accuracy on accepted inputs (%)",
           title="Selective prediction: refusing doubtful inputs")
    ax.legend()
    _save(fig, path)


def sample_gallery(images, results, titles, path, ncols=6):
    """images: (B,1,28,28) in [0,1]; results: list from inference.doubt."""
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 1.9, nrows * 3.0),
                             gridspec_kw={"height_ratios": [1.5, 0.8] * nrows}, squeeze=False)
    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ai, ab = axes[2 * r, c], axes[2 * r + 1, c]
        if k >= n:
            ai.axis("off"); ab.axis("off")
            continue
        res = results[k]
        ai.imshow(images[k, 0], cmap="gray")
        ai.set_title(f"{titles[k]}\npred {res['pred']}  conf {res['confidence']:.2f}\n"
                     f"H {res['entropy']:.2f}  MI {res['mutual_info']:.2f}", fontsize=7)
        ai.axis("off")
        s = res["samples"]
        ab.bar(range(10), s.mean(0), yerr=s.std(0), color=BLUE, ecolor=ORANGE, capsize=1.5)
        ab.set_ylim(0, 1.05)
        ab.set_xticks(range(10))
        ab.tick_params(labelsize=6)
        ab.grid(False)
    _save(fig, path)


def headline_bars(h, path):
    """Two-panel comparison with 95% intervals, in the style of the reference paper.
    h: {method: {"auc": (mean, lo, hi), "tpr": (mean, lo, hi)}}"""
    order = sorted(h, key=lambda m: -h[m]["auc"][0])[::-1]          # best at the top
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    panels = [("auc", "OOD ranked above real digits (AUROC)", "AUROC (0.5 = chance)", 1.0, (0.5, 1.03)),
              ("tpr", "OOD caught at 5% false alarms", "detection rate (%)", 100.0, (0, 112))]
    for ax, (key, title, xl, sc, xlim) in zip(axes, panels):
        for i, m in enumerate(order):
            mean, lo, hi = h[m][key]
            col = BLUE if m.startswith("MC Dropout") else GREY
            left = 0.5 if key == "auc" else 0.0
            ax.barh(i, mean * sc - left, left=left, color=col, height=0.55,
                    xerr=[[max(0, (mean - lo) * sc)], [max(0, (hi - mean) * sc)]],
                    ecolor="black", capsize=3)
            ax.text(hi * sc + (0.012 if key == "auc" else 2.5), i,
                    f"{mean * sc:.3f}" if key == "auc" else f"{mean * sc:.1f}%",
                    va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(order)), order)
        ax.set(xlim=xlim, xlabel=xl)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", visible=False)
    _save(fig, path)


def t_ablation(Ts, auc_mi, auc_ent, ece, nll, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(Ts, auc_mi, "o-", color=ORANGE, label="mutual information")
    ax[0].plot(Ts, auc_ent, "s-", color=BLUE, label="entropy")
    ax[0].set(title="OOD detection (mean AUROC)", xlabel="MC passes T")
    ax[0].legend()
    ax[1].plot(Ts, np.array(ece) * 100, "o-", color=GREEN)
    ax[1].set(title="Calibration error ECE (%)", xlabel="MC passes T")
    ax[2].plot(Ts, nll, "o-", color=RED)
    ax[2].set(title="Negative log-likelihood", xlabel="MC passes T")
    _save(fig, path)
