import numpy as np
from scipy.special import softmax
from src.evaluate import ece, fit_temperature, selective, paired_gap
from src.data import LOCALES


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.full(1000, 0.7)
    correct = np.r_[np.ones(700), np.zeros(300)]
    assert ece(conf, correct) < 1e-9


def test_ece_measures_overconfidence():
    conf = np.full(1000, 0.95)
    correct = np.r_[np.ones(600), np.zeros(400)]
    assert abs(ece(conf, correct) - 0.35) < 1e-9


def test_temperature_recovers_a_known_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (20000, 10))          # true logits
    y = np.array([rng.choice(10, p=p) for p in softmax(z, axis=1)])
    T = fit_temperature(3.0 * z, y)            # model is 3x too sharp
    assert abs(T - 3.0) < 0.15


def test_full_coverage_is_plain_accuracy_and_deferring_helps_when_confidence_is_informative():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=5000)
    correct = (rng.uniform(size=5000) < conf).astype(float)
    s = selective(conf, correct)
    assert abs(s["1.0"] - correct.mean()) < 1e-12
    assert s["1.0"] < s["0.8"] < s["0.5"]


def _run(n, acc, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for loc in LOCALES:
        y = rng.integers(0, 60, n)
        x = rng.normal(size=(n, 60))
        hit = rng.uniform(size=n) < acc
        x[np.arange(n), y] += np.where(hit, 100.0, -100.0)
        out[loc] = (x, y)
    return out


def test_paired_gap_is_zero_for_identical_runs_and_signed_otherwise():
    a = _run(500, 0.8, 0)
    g = paired_gap(a, a)
    assert g["gap"] == 0 and g["ci"] == [0.0, 0.0]
    better = _run(500, 0.9, 1)
    g = paired_gap(better, a)
    assert g["ci"][0] > 0
