import numpy as np
import pytest

from doubtnet import metrics as M


def onehot_mc(classes, T=10, C=10):
    p = np.full((T, len(classes), C), 1e-9)
    p[:, np.arange(len(classes)), classes] = 1.0
    return p / p.sum(-1, keepdims=True)


def test_certain_and_agreeing_passes_have_zero_uncertainty():
    s = M.summarize(onehot_mc([3, 7]))
    assert np.allclose(s["entropy"], 0, atol=1e-6)
    assert np.allclose(s["mutual_info"], 0, atol=1e-6)
    assert np.allclose(s["variation_ratio"], 0)
    assert list(s["pred"]) == [3, 7]


def test_uniform_passes_are_pure_aleatoric():
    mc = np.full((20, 1, 10), 0.1)
    assert np.isclose(M.predictive_entropy(mc)[0], np.log(10))
    assert np.isclose(M.expected_entropy(mc)[0], np.log(10))
    assert np.isclose(M.mutual_information(mc)[0], 0, atol=1e-9)


def test_disagreeing_confident_passes_are_pure_epistemic():
    # half the passes say "1" with certainty, half say "2"
    mc = np.zeros((10, 1, 10)) + 1e-12
    mc[:5, 0, 1] = 1.0
    mc[5:, 0, 2] = 1.0
    assert np.isclose(M.expected_entropy(mc)[0], 0, atol=1e-6)
    assert np.isclose(M.mutual_information(mc)[0], np.log(2), atol=1e-6)
    assert np.isclose(M.variation_ratio(mc)[0], 0.5)


def test_decomposition_adds_up():
    rng = np.random.default_rng(0)
    mc = rng.dirichlet(np.ones(10), size=(30, 50))
    assert np.allclose(M.predictive_entropy(mc),
                       M.expected_entropy(mc) + M.mutual_information(mc))
    assert (M.mutual_information(mc) >= 0).all()


def test_calibration_metrics():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1, 1, 1])
    assert M.accuracy(probs, y) == 0.75
    assert np.isclose(M.nll(probs, y), -np.mean(np.log([0.9, 0.8, 0.4, 0.7])))
    assert np.isclose(M.brier(probs, y), np.mean([0.02, 0.08, 0.72, 0.18]))
    perfect = np.eye(2)[y]
    assert M.ece(perfect, y) < 1e-9


def test_ood_scores_perfect_and_chance():
    assert M.ood_scores(np.zeros(100), np.ones(100))["auroc"] == 1.0
    rng = np.random.default_rng(1)
    r = M.ood_scores(rng.random(5000), rng.random(5000))
    assert 0.47 < r["auroc"] < 0.53


def test_rejection_curve_improves_when_uncertainty_is_informative():
    unc = np.linspace(0, 1, 1000)
    correct = unc < 0.9                      # all errors are the most uncertain 10%
    f, a = M.rejection_curve(unc, correct)
    assert np.isclose(a[0], 0.9)
    assert a[np.argmin(np.abs(f - 0.1))] == 1.0
    assert np.all(np.diff(a) >= -1e-12)


def test_model_shapes_and_mc_dropout_is_stochastic():
    torch = pytest.importorskip("torch")
    from doubtnet.model import DoubtNet, enable_mc_dropout
    m = DoubtNet()
    x = torch.randn(4, 1, 28, 28)
    assert m(x).shape == (4, 10)
    m.eval()
    assert torch.allclose(m(x), m(x))                 # deterministic in eval mode
    enable_mc_dropout(m)
    assert not torch.allclose(m(x), m(x))             # stochastic with MC dropout


def test_tpr_at_fpr_holm_and_brier_skill():
    rng = np.random.default_rng(0)
    a, b = rng.random(10000), rng.random(10000) + 1.0
    assert M.tpr_at_fpr(a, b, 0.05) == 1.0
    assert abs(M.tpr_at_fpr(a, a.copy(), 0.05) - 0.05) < 0.01
    adj = M.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    y = np.array([0, 1, 2, 0, 1, 2])
    assert M.brier_skill(np.eye(3)[y], y) == 1.0
    assert M.brier_skill(np.full((6, 3), 1 / 3), y) < 1e-9
