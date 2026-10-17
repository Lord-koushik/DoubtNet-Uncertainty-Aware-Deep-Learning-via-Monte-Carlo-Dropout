# DoubtNet: Uncertainty-Aware Deep Learning via Monte Carlo Dropout

**Probabilistic reasoning for uncertainty-aware image classification using MNIST and out-of-distribution inputs.**

A normal neural network always answers, and it often answers *confidently* even when shown a photo of a shoe instead of a digit. DoubtNet keeps dropout switched on at test time, runs each image through the network **T** times, and treats the spread of those answers as a measure of doubt. That lets it say *"this is a 7"*, *"this could be a 4 or a 9, ask a human"*, or *"I have never seen anything like this"*.

## 1. Theory in one page

**MC Dropout as Bayesian inference.** Gal & Ghahramani (2016) showed that a network trained with dropout and L2 weight decay is approximately performing variational inference over its weights. Each forward pass with a fresh dropout mask samples weights ω̂ₜ from the approximate posterior q(ω). The predictive distribution is estimated by averaging:

p(y | x, D) ≈ (1/T) Σₜ softmax(f(x; ω̂ₜ))

**Three uncertainty measures** (computed in `doubtnet/metrics.py`):

| Quantity | Formula | Meaning |
|---|---|---|
| Predictive entropy (total) | H[ p̄ ] = −Σ_c p̄_c log p̄_c | everything the model is unsure about |
| Expected entropy (aleatoric) | (1/T) Σₜ H[ pₜ ] | noise in the input itself, e.g. a sloppy 4 that looks like a 9 |
| Mutual information (epistemic) | H[ p̄ ] − (1/T) Σₜ H[ pₜ ] | how much the sampled networks *disagree*; high on unfamiliar inputs |

plus the **variation ratio** (fraction of passes that disagree with the majority vote).

Aleatoric uncertainty does not go away with more data; epistemic uncertainty does, which is why mutual information is the natural signal for out-of-distribution (OOD) detection.

## 2. What the project does

| Stage | What is measured | Output |
|---|---|---|
| Training | CNN (4 conv + 2 FC, ~0.87 M params) with dropout 0.25 / 0.5 on MNIST | `checkpoints/doubtnet.pt`, `figures/training_curves.png` |
| Calibration | Accuracy, NLL, Brier score, ECE; deterministic vs MC; effect of T | `figures/reliability.png` |
| OOD detection | AUROC, AUPR, FPR@95%TPR on Fashion-MNIST, KMNIST, inverted MNIST, uniform & Gaussian noise, blank images | `figures/ood_auroc.png`, `figures/hist_*.png` |
| Dataset shift | MNIST rotated 0°→180°: accuracy falls, does uncertainty rise? | `figures/rotation_shift.png` |
| Selective prediction | Accuracy after refusing the most uncertain 5/10/20 % of inputs | `figures/rejection.png` |
| Decision rule | Thresholds set at the 95th percentile of validation entropy and MI → CONFIDENT / DOUBTFUL / REJECT | `results/report.md` |
| Gallery | Easiest and hardest MNIST digits plus one image from each OOD set, with per-pass probability bars | `figures/gallery.png` |

Every number is written to `results/metrics.json` and a ready-to-paste `results/report.md`.

## 3. Project layout

```
doubtnet/
├── doubtnet/
│   ├── model.py        # DoubtNet CNN + enable_mc_dropout()
│   ├── data.py         # MNIST split, OOD suite, rotated MNIST
│   ├── training.py     # training loop
│   ├── experiments.py  # full study: seeds, ensemble baseline, bootstrap CIs, report
│   ├── inference.py    # deterministic & MC prediction, verdict()
│   ├── metrics.py      # entropy / MI / ECE / AUROC / rejection curves (pure NumPy)
│   ├── preprocess.py   # make any photo look like an MNIST digit
│   └── plots.py        # all figures
├── check_env.py        # run first: diagnoses install / download problems
├── run_all.py          # ONE command: train 5 nets + run every experiment
├── train.py            # train a single network
├── predict.py          # CLI: python predict.py my_digit.png
├── tests/test_metrics.py
└── requirements.txt
```

## 4. How to run

```bash
cd doubtnet
pip3 install -r requirements.txt

python3 check_env.py        # 1. diagnoses problems and tells you the fix
python3 run_all.py --quick  # 2. 3-5 min smoke test of the whole pipeline
python3 run_all.py          # 3. full study: 3 networks x 6 epochs, T=30 (~15-25 min)

python3 predict.py my_digit.png --plot
python3 -m pytest -q        # unit tests
```

`run_all.py` trains 3 independently seeded networks (they double as the **deep-ensemble baseline**), then evaluates
everything: calibration, OOD detection (mean ± std over seeds plus bootstrap 95% intervals), rotation shift, selective
prediction and a gallery. Re-running skips networks that already exist, so it is safe to stop and resume.
Outputs: `results/report.md`, `results/summary.txt`, `results/metrics.json`, `figures/*.png`, and `results_pack.zip` (everything, ready to share).

| Paper element | Produced by `run_all.py` |
|---|---|
| Headline comparison with 95% intervals | `figures/headline.png`, report §7 |
| Calibration incl. Brier skill score | report §1, `figures/reliability.png` |
| OOD detection, mean ± std over seeds | report §2, `figures/ood_auroc.png`, `hist_*.png` |
| Significance tests (paired t-test, Holm) | report §8 |
| Ablation over MC passes T | report §9, `figures/T_ablation.png` |
| Worked examples and failure cases | `figures/gallery.png`, `figures/missed_ood.png`, report §6 and §10 |
| Rotation shift, selective prediction | report §4-5 |

**Troubleshooting**
* First `import torch` takes 30-90 s on a Mac - wait, do not press Ctrl+C.
* `CERTIFICATE_VERIFY_FAILED` while downloading MNIST (macOS python.org Python): open Applications -> Python 3.x ->
  double-click *Install Certificates.command*, then re-run.

## 5. What to expect

These are the qualitative patterns MC Dropout is known to produce on this setup; your exact numbers will be in `results/report.md`.

* MNIST accuracy around 99 %+, with MC averaging matching or slightly improving accuracy and usually lowering NLL and ECE versus single-pass inference.
* The deterministic network still gives **high softmax confidence to many OOD images** (Fashion-MNIST, noise) - the core problem this project addresses.
* MC entropy and mutual information separate MNIST from far-OOD sets much better than chance; noise and blank images are typically easiest, KMNIST and inverted MNIST harder.
* Rotation: accuracy drops sharply after ~45°, and mean entropy / MI climb with it.
* Refusing the most uncertain few percent of test digits pushes accuracy on the remainder close to 100 %.
* **Known limitation:** MC Dropout is a cheap approximation. It can underestimate uncertainty, and some OOD inputs (especially "digit-like" ones) still receive confident wrong answers. Deep ensembles or more expressive posteriors are common comparisons in the literature.

## 6. Ideas for extending it

1. Compare against a 5-model deep ensemble (Lakshminarayanan et al., 2017).
2. Add temperature scaling as a non-Bayesian calibration baseline.
3. Sweep dropout rate p and show the accuracy / uncertainty-quality trade-off.
4. Use mutual information for active learning: label the most-uncertain digits first (BALD).

## References

1. Y. Gal & Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", ICML 2016.
2. A. Kendall & Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017.
3. S. Depeweg et al., "Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning", ICML 2018.
4. D. Hendrycks & K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017.
5. C. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
6. B. Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", NeurIPS 2017.
7. N. Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning", 2011 (BALD).
