"""Training loop shared by train.py and run_all.py."""
import os
import time

import torch
import torch.nn as nn

from .data import mnist_train_val, loader
from .model import DoubtNet, count_parameters, get_device


def run_epoch(model, dl, device, opt=None):
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    tot_loss, tot_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(y)
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, tot_correct / n


def train_model(seed=0, epochs=6, out="checkpoints/seed0.pt", data_root="data",
                batch_size=256, lr=2e-3, weight_decay=1e-4, p_conv=0.25, p_fc=0.5,
                augment=False, device="auto", verbose=True):
    """Train one DoubtNet and save the checkpoint with the best validation loss."""
    torch.manual_seed(seed)
    dev = get_device(device)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    train_ds, val_ds = mnist_train_val(data_root, augment=augment)
    train_dl = loader(train_ds, batch_size, shuffle=True)
    val_dl = loader(val_ds, 512)

    model = DoubtNet(p_conv=p_conv, p_fc=p_fc).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if verbose:
        print(f"  seed {seed}: {count_parameters(model):,} params | device={dev} | "
              f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best = float("inf")
    tmp = out + ".partial"          # only renamed to `out` when training finishes completely
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, dev, opt)
        vl, va = run_epoch(model, val_dl, dev)
        sched.step()
        for k, v in zip(hist, [tl, ta, vl, va]):
            hist[k].append(v)
        saved = ""
        if vl < best:
            best = vl
            torch.save({"state_dict": model.state_dict(),
                        "config": {"p_conv": p_conv, "p_fc": p_fc, "seed": seed, "epochs": epochs},
                        "epoch": ep, "val_loss": vl, "val_acc": va}, tmp)
            saved = " *"
        if verbose:
            print(f"  seed {seed} epoch {ep}/{epochs}  train {ta * 100:.2f}%  val {va * 100:.2f}% "
                  f"(loss {vl:.4f})  {time.time() - t0:.0f}s{saved}", flush=True)
    os.replace(tmp, out)            # a finished model; an interrupted run leaves no half-trained file
    return hist
