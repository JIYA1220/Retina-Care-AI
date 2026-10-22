import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import bootstrap_ci, ece, fit_temperature, nll, qwk, softmax  # noqa: E402
from utils.preprocess import ben_graham_preprocess  # noqa: E402


def test_qwk_perfect_agreement_is_one():
    y = np.array([0, 1, 2, 3, 4, 2, 2])
    assert qwk(y, y) == 1.0


def test_qwk_swapped_two_samples_is_minus_one():
    # derived by hand: observed disagreement 1/16, expected 0.5/16 -> 1 - 2 = -1
    assert np.isclose(qwk([0, 1], [1, 0]), -1.0)


def test_qwk_all_off_by_one_is_five_sevenths():
    # derived by hand: observed 1/16, expected 56/256 -> 1 - 0.0625/0.21875 = 5/7
    assert np.isclose(qwk([0, 1, 2, 3], [1, 2, 3, 4]), 5 / 7)


def test_ece_known_value():
    # conf 0.9: half right -> |0.5-0.9|*0.5 = 0.2 ; conf 0.7: all right -> |1-0.7|*0.5 = 0.15
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.7, 0.3], [0.7, 0.3]])
    y = np.array([0, 1, 0, 0])
    assert np.isclose(ece(probs, y, n_bins=15), 0.35)


def test_ece_zero_when_confident_and_always_right():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, np.array([0, 1])) == 0.0


def test_nll_known_value():
    probs = np.array([[0.5, 0.5], [0.25, 0.75]])
    assert np.isclose(nll(probs, np.array([0, 1])), -(np.log(0.5) + np.log(0.75)) / 2)


def test_temperature_fit_lowers_nll_on_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    logits = rng.normal(0, 1, (400, 5))
    logits[np.arange(400), y] += 1.0          # informative but noisy
    overconfident = logits * 5
    T = fit_temperature(overconfident, y)
    assert T > 1.0
    assert nll(softmax(overconfident / T), y) < nll(softmax(overconfident), y)


def test_bootstrap_ci_is_deterministic_and_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 200)
    pred = np.clip(y + rng.integers(-1, 2, 200), 0, 4)
    a = bootstrap_ci(y, pred, n_resamples=200, seed=0)
    b = bootstrap_ci(y, pred, n_resamples=200, seed=0)
    assert a == b
    assert a[0] <= qwk(y, pred) <= a[1]


def test_preprocess_output_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    out = ben_graham_preprocess(img, img_size=160)
    assert out.shape == (160, 160, 3)
    assert out.dtype == np.uint8
