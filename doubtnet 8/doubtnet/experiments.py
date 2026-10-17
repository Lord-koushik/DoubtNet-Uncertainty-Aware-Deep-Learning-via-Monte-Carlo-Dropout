"""The full experimental study.

Compares, over K independently trained networks (seeds):
  * Softmax confidence of one network              (the standard baseline)
  * MC Dropout predictive entropy / mutual information   (our method)
  * Deep ensemble entropy / mutual information     (strong reference method)
on calibration, OOD detection, rotation shift and selective prediction.
Results are mean +- std over seeds, with bootstrap 95% intervals for AUROC.
Every stage is isolated: if one fails the others still run.
"""
import glob
import json
import os
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

from . import data, metrics, plots
from .inference import predict_deterministic, predict_mc, doubt, verdict
from .model import load_model, get_device

M_SOFT = "Softmax (1 net)"
M_ENT = "MC Dropout - entropy"
M_MI = "MC Dropout - MI"
M_EENT = "Ensemble - entropy"
M_EMI = "Ensemble - MI"


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def fmt(v, scale=100.0, d=2):
    m, s = ms(v)
    return f"{m * scale:.{d}f} ± {s * scale:.{d}f}" if len(v) > 1 else f"{m * scale:.{d}f}"


def stage(name, fn):
    print(f"[{name}]", flush=True)
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        print(f"   !! stage '{name}' failed - continuing with the rest", flush=True)
        return None


def _collect(model, dl, device, T):
    det, y = predict_deterministic(model, dl, device)
    mc, _ = predict_mc(model, dl, device, T)
    s = metrics.summarize(mc)
    return {"det": det, "y": y, "mc_mean": s["mean_probs"], "mc_ent": s["entropy"], "mc_mi": s["mutual_info"]}


def run(ckpt_dir="checkpoints", T=30, n_ood=2000, data_root="data", device_name="auto",
        n_boot=200, seed=0):
    device = get_device(device_name)
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "seed*.pt")))
    if not paths:
        raise SystemExit("No checkpoints/seed*.pt found. Train first:  python3 run_all.py")
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    models = [load_model(p, device) for p in paths]
    K = len(models)
    ens_ok = K >= 2
    R = {"T": T, "n_ood": n_ood, "n_models": K}

    test_ds = data.mnist_test(data_root)
    oods = data.ood_datasets(data_root, n=n_ood, seed=seed)
    sets = {"MNIST": test_ds, **oods}

    # ------------------------------------------------------------------ collect
    C = []
    for k, m in enumerate(models):
        print(f"[collect] model {k + 1}/{K}: MC Dropout (T={T}) on {len(sets)} datasets", flush=True)
        C.append({n: _collect(m, data.loader(ds), device, T) for n, ds in sets.items()})
    y = C[0]["MNIST"]["y"]

    ens = {}
    if ens_ok:
        for n in sets:
            s = metrics.summarize(np.stack([C[k][n]["det"] for k in range(K)]))
            ens[n] = {"mean": s["mean_probs"], "ent": s["entropy"], "mi": s["mutual_info"]}

    def scores(k, n):
        c = C[k][n]
        return {M_SOFT: 1.0 - c["det"].max(-1), M_ENT: c["mc_ent"], M_MI: c["mc_mi"]}

    def ens_scores(n):
        return {M_EENT: ens[n]["ent"], M_EMI: ens[n]["mi"]}

    methods = [M_SOFT, M_ENT, M_MI] + ([M_EENT, M_EMI] if ens_ok else [])

    # ------------------------------------------------------------- calibration
    def calibration():
        rows = {"Deterministic (dropout off)": [], "MC Dropout": []}
        for k in range(K):
            rows["Deterministic (dropout off)"].append(metrics.calibration_report(C[k]["MNIST"]["det"], y))
            rows["MC Dropout"].append(metrics.calibration_report(C[k]["MNIST"]["mc_mean"], y))
        out = {n: {m: [r[m] for r in rs] for m in rs[0]} for n, rs in rows.items()}
        if ens_ok:
            out["Deep ensemble"] = {m: [v] for m, v in metrics.calibration_report(ens["MNIST"]["mean"], y).items()}
        R["calibration"] = out
        plots.reliability_diagram(C[0]["MNIST"]["det"], C[0]["MNIST"]["mc_mean"], y, "figures/reliability.png")
    stage("calibration", calibration)

    # ------------------------------------------------------------ OOD detection
    ood_names = list(oods)
    ood_res = {}

    def ood():
        for o in ood_names:
            ood_res[o] = {m: {"auroc": [], "fpr95": []} for m in methods}
            for k in range(K):
                sin, sout = scores(k, "MNIST"), scores(k, o)
                for m in (M_SOFT, M_ENT, M_MI):
                    r = metrics.ood_scores(sin[m], sout[m])
                    ood_res[o][m]["auroc"].append(r["auroc"])
                    ood_res[o][m]["fpr95"].append(r["fpr_at_95tpr"])
            if ens_ok:
                ein, eout = ens_scores("MNIST"), ens_scores(o)
                for m in (M_EENT, M_EMI):
                    r = metrics.ood_scores(ein[m], eout[m])
                    ood_res[o][m]["auroc"].append(r["auroc"])
                    ood_res[o][m]["fpr95"].append(r["fpr_at_95tpr"])
            sm = ood_res[o][M_SOFT]["auroc"]
            print(f"   {o:15s} AUROC softmax {np.mean(sm):.3f} | MC entropy "
                  f"{np.mean(ood_res[o][M_ENT]['auroc']):.3f} | MC MI {np.mean(ood_res[o][M_MI]['auroc']):.3f}",
                  flush=True)
        R["ood"] = ood_res
        R["mean_det_confidence"] = {o: float(np.mean([C[k][o]["det"].max(-1).mean() for k in range(K)]))
                                    for o in ood_names}
        plots.ood_auroc_bars({o: {m: float(np.mean(ood_res[o][m]["auroc"])) for m in methods}
                              for o in ood_names}, "figures/ood_auroc.png")
        plots.uncertainty_histograms(C[0]["MNIST"]["mc_ent"], {o: C[0][o]["mc_ent"] for o in ood_names},
                                     "Predictive entropy", "figures/hist_entropy.png")
        plots.uncertainty_histograms(C[0]["MNIST"]["mc_mi"], {o: C[0][o]["mc_mi"] for o in ood_names},
                                     "Mutual information", "figures/hist_mutual_info.png")
    stage("OOD detection", ood)

    # --------------------------------------------------------- bootstrap CIs
    def boot():
        out = {}
        for o in ood_names:
            sin, sout = scores(0, "MNIST"), scores(0, o)
            out[o] = {m: metrics.bootstrap_auroc(sin[m], sout[m], n_boot) for m in (M_SOFT, M_ENT, M_MI)}
            if ens_ok:
                ein, eout = ens_scores("MNIST"), ens_scores(o)
                out[o][M_EMI] = metrics.bootstrap_auroc(ein[M_EMI], eout[M_EMI], n_boot)
        R["bootstrap_ci_seed0"] = out
    stage("bootstrap intervals", boot)

    # ------------------------------------------------- thresholds & flag rates
    def flags():
        _, val_ds = data.mnist_train_val(data_root)
        vmc, _ = predict_mc(models[0], data.loader(val_ds), device, T)
        vs = metrics.summarize(vmc)
        th = {"entropy": metrics.threshold_at_coverage(vs["entropy"], 0.95),
              "mutual_info": metrics.threshold_at_coverage(vs["mutual_info"], 0.95)}
        R["thresholds"] = th
        R["flag_rate"] = {n: float(((C[0][n]["mc_ent"] > th["entropy"]) |
                                    (C[0][n]["mc_mi"] > th["mutual_info"])).mean()) for n in sets}
    stage("decision thresholds", flags)

    # ----------------------------------------------------------- rotation shift
    def rotation():
        angles = list(range(0, 181, 15))
        rot = {"angle": angles, "acc_det": [], "acc_mc": [], "ece_mc": [], "entropy": [], "mutual_info": []}
        for a in angles:
            dl = data.loader(data.rotated_mnist(a, data_root, n=n_ood, seed=seed))
            d, yr = predict_deterministic(models[0], dl, device)
            mc, _ = predict_mc(models[0], dl, device, T)
            s = metrics.summarize(mc)
            rot["acc_det"].append(metrics.accuracy(d, yr))
            rot["acc_mc"].append(metrics.accuracy(s["mean_probs"], yr))
            rot["ece_mc"].append(metrics.ece(s["mean_probs"], yr))
            rot["entropy"].append(float(s["entropy"].mean()))
            rot["mutual_info"].append(float(s["mutual_info"].mean()))
        R["rotation"] = rot
        plots.rotation_curve(angles, rot["acc_mc"], rot["entropy"], rot["mutual_info"], "figures/rotation_shift.png")
    stage("rotation shift", rotation)

    # ------------------------------------------------------ selective prediction
    def selective():
        c0 = C[0]["MNIST"]
        sc = scores(0, "MNIST")
        right_det = c0["det"].argmax(-1) == y
        right_mc = c0["mc_mean"].argmax(-1) == y
        curves = {M_SOFT: metrics.rejection_curve(sc[M_SOFT], right_det),
                  M_ENT: metrics.rejection_curve(sc[M_ENT], right_mc),
                  M_MI: metrics.rejection_curve(sc[M_MI], right_mc)}
        if ens_ok:
            right_e = ens["MNIST"]["mean"].argmax(-1) == y
            curves[M_EENT] = metrics.rejection_curve(ens["MNIST"]["ent"], right_e)
        R["rejection"] = {k: {"fraction_refused": f, "accuracy": a} for k, (f, a) in curves.items()}
        plots.rejection_plot(curves, "figures/rejection.png")
    stage("selective prediction", selective)

    # ------------------------------------------------------------------ gallery
    def gallery():
        import torch
        ent = C[0]["MNIST"]["mc_ent"]
        imgs, titles = [], []
        for i in np.argsort(ent)[:3]:
            imgs.append(test_ds[int(i)][0]); titles.append(f"MNIST true {y[i]} (easy)")
        for i in np.argsort(ent)[-9:][::-1]:
            imgs.append(test_ds[int(i)][0]); titles.append(f"MNIST true {y[i]} (hard)")
        for n in ["Fashion-MNIST", "KMNIST", "Inverted MNIST", "Uniform noise", "Gaussian noise", "Blank"]:
            imgs.append(oods[n][0][0]); titles.append(n)
        batch = torch.stack(imgs)
        res = doubt(models[0], batch, device, T)
        plots.sample_gallery(data.denormalize(batch).numpy(), res, titles, "figures/gallery.png")
        th = R.get("thresholds", {"entropy": 0.5, "mutual_info": 0.05})
        R["gallery"] = [{"title": t, "verdict": verdict(r, th),
                         **{k: r[k] for k in ["pred", "confidence", "entropy", "mutual_info"]}}
                        for t, r in zip(titles, res)]
    stage("gallery", gallery)


    # ------------------------------------------- headline comparison (+ 95% CIs)
    def headline():
        rng = np.random.default_rng(seed)
        acc = {m: {"auc": [], "tpr": []} for m in methods}
        n_in = len(C[0]["MNIST"]["y"])
        for _ in range(n_boot):
            k = int(rng.integers(K))                 # resample the network AND the inputs
            ii = rng.integers(0, n_in, n_in)
            sin = scores(k, "MNIST")
            if ens_ok:
                sin.update(ens_scores("MNIST"))
            per = {m: ([], []) for m in methods}
            for o in ood_names:
                n_out = len(C[0][o]["y"])
                jj = rng.integers(0, n_out, n_out)
                sout = scores(k, o)
                if ens_ok:
                    sout.update(ens_scores(o))
                for m in methods:
                    a, b = sin[m][ii], sout[m][jj]
                    per[m][0].append(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
                    per[m][1].append(metrics.tpr_at_fpr(a, b, 0.05))
            for m in methods:
                acc[m]["auc"].append(np.mean(per[m][0]))
                acc[m]["tpr"].append(np.mean(per[m][1]))
        out = {m: {key: (float(np.mean(v)), *[float(q) for q in np.quantile(v, [0.025, 0.975])])
                   for key, v in d.items()} for m, d in acc.items()}
        R["headline"] = out
        plots.headline_bars(out, "figures/headline.png")
        for m in methods:
            print(f"   {m:26s} AUROC {out[m]['auc'][0]:.4f} [{out[m]['auc'][1]:.4f}, {out[m]['auc'][2]:.4f}]"
                  f"   TPR@5%FPR {out[m]['tpr'][0] * 100:.1f}%", flush=True)
    stage("headline comparison", headline)

    # ----------------------------------------------- paired significance tests
    def stats_tests():
        from scipy import stats as st
        if K < 3:
            print("   need >= 3 seeds for paired tests; skipped")
            return
        rows = []
        for m in (M_MI, M_ENT):
            for o in ood_names:
                a, b = ood_res[o][m]["auroc"], ood_res[o][M_SOFT]["auroc"]
                p = float(np.nan_to_num(st.ttest_rel(a, b).pvalue, nan=1.0))
                rows.append({"method": m, "ood": o, "diff": float(np.mean(a) - np.mean(b)), "p": p})
        for r, q in zip(rows, metrics.holm([r["p"] for r in rows])):
            r["p_holm"] = q
        R["paired_tests"] = rows
    stage("paired tests vs softmax", stats_tests)

    # ------------------------------------------------------ ablation over T
    def t_ablation():
        grid = sorted({t for t in (1, 2, 3, 5, 10, 20, 30, 50) if t <= T} | {T})
        id_dl = data.loader(data.head(test_ds, 2000))
        mc_id, y_id = predict_mc(models[0], id_dl, device, T)
        mc_ood = [predict_mc(models[0], data.loader(oods[o]), device, T)[0] for o in ("Fashion-MNIST", "KMNIST")]
        ab = {"T": grid, "auroc_mi": [], "auroc_ent": [], "ece": [], "nll": [], "acc": []}
        for t in grid:
            a, e = [], []
            for mo in mc_ood:
                y_ = np.r_[np.zeros(mc_id.shape[1]), np.ones(mo.shape[1])]
                a.append(roc_auc_score(y_, np.r_[metrics.mutual_information(mc_id[:t]), metrics.mutual_information(mo[:t])]))
                e.append(roc_auc_score(y_, np.r_[metrics.predictive_entropy(mc_id[:t]), metrics.predictive_entropy(mo[:t])]))
            pm = mc_id[:t].mean(0)
            ab["auroc_mi"].append(float(np.mean(a))); ab["auroc_ent"].append(float(np.mean(e)))
            ab["ece"].append(metrics.ece(pm, y_id)); ab["nll"].append(metrics.nll(pm, y_id))
            ab["acc"].append(metrics.accuracy(pm, y_id))
        R["T_ablation"] = ab
        plots.t_ablation(grid, ab["auroc_mi"], ab["auroc_ent"], ab["ece"], ab["nll"], "figures/T_ablation.png")
    stage("ablation over MC passes T", t_ablation)

    # ---------------------------------- failure cases: OOD inputs NOT flagged
    def missed():
        import torch
        cand = []
        for o in ("Fashion-MNIST", "KMNIST", "Inverted MNIST"):
            ent = C[0][o]["mc_ent"]
            cand += [(o, int(i), float(ent[i])) for i in np.argsort(ent)[:2]]
        cand = sorted(cand, key=lambda t: t[2])[:6]
        batch = torch.stack([oods[o][i][0] for o, i, _ in cand])
        res = doubt(models[0], batch, device, T)
        titles = [f"{o} #{i}" for o, i, _ in cand]
        plots.sample_gallery(data.denormalize(batch).numpy(), res, titles, "figures/missed_ood.png", ncols=3)
        th = R.get("thresholds", {"entropy": 0.5, "mutual_info": 0.05})
        R["missed_ood"] = [{"title": t, "verdict": verdict(r, th),
                            **{k: r[k] for k in ["pred", "confidence", "entropy", "mutual_info"]}}
                           for t, r in zip(titles, res)]
    stage("failure cases (missed OOD)", missed)

    with open("results/metrics.json", "w") as f:
        json.dump(_py(R), f, indent=2)
    write_report(R, methods)
    print("\nDONE -> results/report.md, results/metrics.json, figures/*.png", flush=True)
    return R


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_report(R, methods):
    K = R["n_models"]
    L = ["# DoubtNet - results", "",
         f"MC Dropout passes T = **{R['T']}**; {K} independently trained networks (seeds); "
         f"{R['n_ood']} images per OOD set. Values are mean ± std over seeds.", ""]
    if "calibration" in R:
        L += ["## 1. MNIST test set - accuracy and calibration", "",
              "| Method | Accuracy (%) | NLL | Brier | Brier skill | ECE (%) |", "|---|---|---|---|---|---|"]
        for name, m in R["calibration"].items():
            L.append(f"| {name} | {fmt(m['accuracy'])} | {fmt(m['nll'], 1, 4)} | {fmt(m['brier'], 1, 4)} | {fmt(m['brier_skill'], 1, 4)} | {fmt(m['ece'])} |")
        L += ["", "![reliability](../figures/reliability.png)", ""]
    if "ood" in R:
        L += ["## 2. OOD detection - AUROC (%), higher is better, 50 = chance", "",
              "| OOD set | Softmax conf. | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]
        for o, d in R["ood"].items():
            L.append(f"| {o} | {R['mean_det_confidence'][o] * 100:.1f} | " +
                     " | ".join(fmt(d[m]["auroc"]) for m in methods) + " |")
        avg = {m: np.mean([np.mean(R["ood"][o][m]["auroc"]) for o in R["ood"]]) for m in methods}
        L.append("| **Average** | | " + " | ".join(f"**{avg[m] * 100:.2f}**" for m in methods) + " |")
        L += ["", "(Second column: the plain network's mean softmax confidence on that OOD set.)", "",
              "FPR at 95% TPR (%), lower is better:", "",
              "| OOD set | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
        for o, d in R["ood"].items():
            L.append(f"| {o} | " + " | ".join(fmt(d[m]["fpr95"]) for m in methods) + " |")
        L += ["", "![auroc](../figures/ood_auroc.png)", "", "![h1](../figures/hist_entropy.png)", "",
              "![h2](../figures/hist_mutual_info.png)", ""]
    if "bootstrap_ci_seed0" in R:
        L += ["Bootstrap 95% intervals for AUROC (seed 0, %):", "", "| OOD set | Method | AUROC 95% CI |", "|---|---|---|"]
        for o, d in R["bootstrap_ci_seed0"].items():
            for m, (lo, hi) in d.items():
                L.append(f"| {o} | {m} | [{lo * 100:.2f}, {hi * 100:.2f}] |")
        L.append("")
    if "flag_rate" in R:
        th = R["thresholds"]
        L += ["## 3. Decision rule", "",
              f"Flag if entropy > {th['entropy']:.3f} or MI > {th['mutual_info']:.4f} "
              "(95th percentile on the validation split).", "",
              "| Input set | Flagged (%) |", "|---|---|"]
        L += [f"| {n} | {v * 100:.1f} |" for n, v in R["flag_rate"].items()]
        L.append("")
    if "rotation" in R:
        r = R["rotation"]
        L += ["## 4. Dataset shift - rotated MNIST (seed 0)", "",
              "| Angle | Acc det (%) | Acc MC (%) | ECE MC (%) | Entropy | MI |", "|---|---|---|---|---|---|"]
        for i, a in enumerate(r["angle"]):
            L.append(f"| {a} | {r['acc_det'][i] * 100:.2f} | {r['acc_mc'][i] * 100:.2f} | "
                     f"{r['ece_mc'][i] * 100:.2f} | {r['entropy'][i]:.3f} | {r['mutual_info'][i]:.4f} |")
        L += ["", "![rot](../figures/rotation_shift.png)", ""]
    if "rejection" in R:
        L += ["## 5. Selective prediction (seed 0)", "",
              "| Score | Acc @0% refused | @5% | @10% | @20% |", "|---|---|---|---|---|"]
        for k, v in R["rejection"].items():
            f, a = np.array(v["fraction_refused"]), np.array(v["accuracy"])
            pick = [a[np.argmin(np.abs(f - q))] * 100 for q in [0, .05, .10, .20]]
            L.append(f"| {k} | " + " | ".join(f"{p:.2f}" for p in pick) + " |")
        L += ["", "![rej](../figures/rejection.png)", ""]
    if "headline" in R:
        L += ["## 7. Headline comparison (95% intervals)", "",
              "Mean over the six OOD sets; intervals resample both the trained network and the test inputs.", "",
              "| Method | OOD AUROC [95% CI] | OOD caught @5% false alarms (%) [95% CI] |", "|---|---|---|"]
        for m, d in sorted(R["headline"].items(), key=lambda kv: -kv[1]["auc"][0]):
            a, t = d["auc"], d["tpr"]
            L.append(f"| {m} | {a[0]:.4f} [{a[1]:.4f}, {a[2]:.4f}] | {t[0] * 100:.1f} [{t[1] * 100:.1f}, {t[2] * 100:.1f}] |")
        L += ["", "![headline](../figures/headline.png)", ""]
    if "paired_tests" in R:
        L += ["## 8. Paired tests across seeds (AUROC difference vs softmax)", "",
              "| Method | OOD set | AUROC diff | p | p (Holm) |", "|---|---|---|---|---|"]
        for r in R["paired_tests"]:
            L.append(f"| {r['method']} | {r['ood']} | {r['diff'] * 100:+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} |")
        L += ["", "Paired t-test over the trained networks; Holm correction over all comparisons. "
              "With few seeds these tests have limited power.", ""]
    if "T_ablation" in R:
        a = R["T_ablation"]
        L += ["## 9. Ablation: number of MC passes T (seed 0)", "",
              "| T | Accuracy (%) | ECE (%) | NLL | AUROC (MI) | AUROC (entropy) |", "|---|---|---|---|---|---|"]
        for i, t in enumerate(a["T"]):
            L.append(f"| {t} | {a['acc'][i] * 100:.2f} | {a['ece'][i] * 100:.2f} | {a['nll'][i]:.4f} | "
                     f"{a['auroc_mi'][i]:.4f} | {a['auroc_ent'][i]:.4f} |")
        L += ["", "![T](../figures/T_ablation.png)", ""]
    if "missed_ood" in R:
        L += ["## 10. Failure cases: OOD inputs the model was most confident about", "",
              "| Input | Pred | Confidence | Entropy | MI | Verdict |", "|---|---|---|---|---|---|"]
        for g in R["missed_ood"]:
            L.append(f"| {g['title']} | {g['pred']} | {g['confidence']:.3f} | {g['entropy']:.3f} | "
                     f"{g['mutual_info']:.4f} | {g['verdict']} |")
        L += ["", "![missed](../figures/missed_ood.png)", ""]
    if "gallery" in R:
        L += ["## 6. Gallery", "", "| Input | Pred | Conf | Entropy | MI | Verdict |", "|---|---|---|---|---|---|"]
        for g in R["gallery"]:
            L.append(f"| {g['title']} | {g['pred']} | {g['confidence']:.3f} | {g['entropy']:.3f} | "
                     f"{g['mutual_info']:.4f} | {g['verdict']} |")
        L += ["", "![gallery](../figures/gallery.png)", ""]
    with open("results/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def print_summary(R, path="results/summary.txt"):
    lines = ["=" * 64, "SUMMARY (mean ± std over seeds; copy these into the paper)", "=" * 64]
    for name, m in R.get("calibration", {}).items():
        lines.append(f"{name:28s} acc {fmt(m['accuracy'])}%   ECE {fmt(m['ece'])}%   NLL {fmt(m['nll'], 1, 4)}")
    if "headline" in R:
        lines += ["", "OOD detection, mean over 6 OOD sets (95% CI):"]
        for m, d in sorted(R["headline"].items(), key=lambda kv: -kv[1]["auc"][0]):
            a, t = d["auc"], d["tpr"]
            lines.append(f"  {m:26s} AUROC {a[0]:.4f} [{a[1]:.4f}, {a[2]:.4f}]   caught@5%FPR {t[0] * 100:.1f}%")
    if "rejection" in R:
        lines += ["", "Accuracy after refusing the most uncertain 10% (seed 0):"]
        for k, v in R["rejection"].items():
            f, a = np.array(v["fraction_refused"]), np.array(v["accuracy"])
            lines.append(f"  {k:26s} {a[np.argmin(np.abs(f - 0.10))] * 100:.2f}%")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
