"""
metrics.py
Evaluation metrics shared by the scripts in scripts/ and by the tests.
"""

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

NUM_CLASSES = 5


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa over the 5 DR grades."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                   labels=list(range(NUM_CLASSES))))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected calibration error with equal-width confidence bins.
    Confidence is the max probability; accuracy is top-1 correctness.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def bootstrap_ci(y_true, y_pred, metric=qwk, n_resamples: int = 1000,
                 seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of a metric over images (resampled with replacement)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        v = metric(y_true[idx], y_pred[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a single temperature T > 0 by minimising NLL (L-BFGS on log T)."""
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(y_true, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
