"""Deterministic and Monte Carlo Dropout inference."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import enable_mc_dropout
from . import metrics


@torch.no_grad()
def predict_deterministic(model, loader, device):
    """Standard inference: dropout off, one pass. Returns (probs (N,C), labels (N,))."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=-1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def predict_mc(model, loader, device, T=50):
    """MC Dropout: T stochastic passes per batch. Returns (mc_probs (T,N,C), labels (N,))."""
    enable_mc_dropout(model)
    chunks, labels = [], []
    for x, y in loader:
        x = x.to(device)
        chunks.append(torch.stack([F.softmax(model(x), dim=-1) for _ in range(T)]).cpu())
        labels.append(y)
    model.eval()
    return torch.cat(chunks, dim=1).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def doubt(model, images, device, T=50):
    """Predict for a small batch of already-normalised images (B,1,28,28).

    Returns a list of dicts, one per image, with the prediction and every
    uncertainty measure plus the raw MC samples (for plotting).
    """
    enable_mc_dropout(model)
    x = images.to(device)
    mc = torch.stack([F.softmax(model(x), dim=-1) for _ in range(T)]).cpu().numpy()
    model.eval()
    s = metrics.summarize(mc)
    out = []
    for i in range(mc.shape[1]):
        out.append({
            "pred": int(s["pred"][i]),
            "confidence": float(s["confidence"][i]),
            "entropy": float(s["entropy"][i]),
            "aleatoric": float(s["aleatoric"][i]),
            "mutual_info": float(s["mutual_info"][i]),
            "variation_ratio": float(s["variation_ratio"][i]),
            "mean_probs": s["mean_probs"][i],
            "samples": mc[:, i, :],
        })
    return out


def verdict(result, thresholds):
    """Turn numbers into a decision a human can act on."""
    if result["mutual_info"] > thresholds["mutual_info"]:
        return "REJECT - unfamiliar input (high epistemic uncertainty)"
    if result["entropy"] > thresholds["entropy"]:
        return "DOUBTFUL - ambiguous digit, refer to a human"
    return "CONFIDENT"
