"""Uncertainty and calibration metrics. Pure NumPy so they can be unit-tested
without PyTorch.

Shapes: `mc_probs` is (T, N, C) = (MC samples, inputs, classes).

Decomposition used throughout (Depeweg et al., 2018):
    total uncertainty      H[ E_t p_t ]            = predictive entropy
    aleatoric (data noise) E_t H[ p_t ]            = expected entropy
    epistemic (model doubt) H[E_t p_t] - E_t H[p_t] = mutual information (BALD)
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

EPS = 1e-12


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -np.sum(p * np.log(p + EPS), axis=axis)


def predictive_entropy(mc_probs: np.ndarray) -> np.ndarray:
    return entropy(mc_probs.mean(axis=0))


def expected_entropy(mc_probs: np.ndarray) -> np.ndarray:
    return entropy(mc_probs, axis=-1).mean(axis=0)


def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
    return np.clip(predictive_entropy(mc_probs) - expected_entropy(mc_probs), 0.0, None)


def variation_ratio(mc_probs: np.ndarray) -> np.ndarray:
    """1 - (fraction of MC passes that vote for the modal class)."""
    T, N, C = mc_probs.shape
    votes = mc_probs.argmax(axis=-1)                       # (T, N)
    counts = np.stack([(votes == c).sum(axis=0) for c in range(C)], axis=-1)
    return 1.0 - counts.max(axis=-1) / T


def predicted_class_std(mc_probs: np.ndarray) -> np.ndarray:
    """Std-dev across MC passes of the probability given to the final predicted class."""
    pred = mc_probs.mean(axis=0).argmax(axis=-1)
    return mc_probs[:, np.arange(mc_probs.shape[1]), pred].std(axis=0)


def summarize(mc_probs: np.ndarray) -> dict:
    mean = mc_probs.mean(axis=0)
    return {
        "mean_probs": mean,
        "pred": mean.argmax(axis=-1),
        "confidence": mean.max(axis=-1),
        "entropy": predictive_entropy(mc_probs),
        "aleatoric": expected_entropy(mc_probs),
        "mutual_info": mutual_information(mc_probs),
        "variation_ratio": variation_ratio(mc_probs),
        "std": predicted_class_std(mc_probs),
    }


# ----------------------------------------------------------------- calibration
def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(-1) == labels).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + EPS).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(-1).mean())


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Returns (ece, bin_centers, bin_accuracy, bin_confidence, bin_counts)."""
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    acc_b, conf_b, cnt_b = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        cnt_b[b] = m.sum()
        if cnt_b[b]:
            acc_b[b], conf_b[b] = correct[m].mean(), conf[m].mean()
    ece = float(np.sum(cnt_b / len(conf) * np.abs(acc_b - conf_b)))
    centers = (edges[:-1] + edges[1:]) / 2
    return ece, centers, acc_b, conf_b, cnt_b


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    return reliability(probs, labels, n_bins)[0]


def brier_skill(probs: np.ndarray, labels: np.ndarray) -> float:
    """1 - Brier / Brier(reference). Reference = always predict the class frequencies.
    > 0 means better than the reference, < 0 worse."""
    onehot = np.eye(probs.shape[1])[labels]
    prior = onehot.mean(0, keepdims=True)
    ref = float(((onehot - prior) ** 2).sum(-1).mean())
    return 1.0 - brier(probs, labels) / ref


def calibration_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    return {"accuracy": accuracy(probs, labels), "nll": nll(probs, labels),
            "brier": brier(probs, labels), "brier_skill": brier_skill(probs, labels),
            "ece": ece(probs, labels)}


# ----------------------------------------------------------- OOD detection
def ood_scores(in_unc: np.ndarray, out_unc: np.ndarray) -> dict:
    """Higher uncertainty should mean 'out-of-distribution' (positive class = OOD)."""
    y = np.r_[np.zeros(len(in_unc)), np.ones(len(out_unc))]
    s = np.r_[in_unc, out_unc]
    fpr, tpr, _ = roc_curve(y, s)
    # FPR@95%TPR: fraction of in-distribution inputs wrongly flagged when 95% of OOD is caught
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")])
    return {"auroc": float(roc_auc_score(y, s)), "aupr": float(average_precision_score(y, s)),
            "fpr_at_95tpr": fpr95}


# ------------------------------------------------- selective classification
def rejection_curve(uncertainty: np.ndarray, correct: np.ndarray, steps: int = 20):
    """Accuracy on the retained inputs after refusing the most uncertain fraction."""
    order = np.argsort(uncertainty)                  # most certain first
    correct = correct[order].astype(float)
    n = len(correct)
    fractions = np.linspace(0.0, 0.95, steps)
    accs = [correct[: max(1, int(round(n * (1 - f))))].mean() for f in fractions]
    return fractions, np.array(accs)


def threshold_at_coverage(uncertainty: np.ndarray, coverage: float = 0.95) -> float:
    """Uncertainty value below which `coverage` of the (in-distribution) inputs fall."""
    return float(np.quantile(uncertainty, coverage))


def bootstrap_auroc(in_unc, out_unc, n_boot=200, seed=0, alpha=0.05):
    """95% bootstrap interval for OOD AUROC (resamples inputs with replacement)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        a = rng.choice(in_unc, len(in_unc))
        b = rng.choice(out_unc, len(out_unc))
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        vals.append(roc_auc_score(y, np.r_[a, b]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def tpr_at_fpr(in_unc, out_unc, fpr=0.05):
    """Fraction of OOD inputs flagged when the threshold lets only `fpr` of in-distribution through."""
    thr = np.quantile(in_unc, 1.0 - fpr)
    return float((np.asarray(out_unc) > thr).mean())


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
