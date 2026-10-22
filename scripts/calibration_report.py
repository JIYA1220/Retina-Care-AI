"""
calibration_report.py
ECE (15 bins) and NLL before / after temperature scaling on the validation split.

The temperature is never fitted and evaluated on the same images: the split is
divided into two stratified halves (fixed seed); T is fitted on one half and
applied to the other, in both directions, and the held-out predictions are pooled.
For reference the report also includes the shipped temperatures (model/*_temp.pt),
which were fitted on the full validation split, so that column is in-sample.

Needs results/val_logits.npz (run scripts/evaluate.py first).
Outputs: results/calibration.json, results/reliability_diagrams.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

import _common as C
from utils.metrics import ece, fit_temperature, nll, softmax

SEED = 0
N_BINS = 15


def reliability_points(probs, y, n_bins=N_BINS):
    conf, correct = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    path = os.path.join(C.RESULTS, "val_logits.npz")
    if not os.path.exists(path):
        sys.exit("results/val_logits.npz not found. Run scripts/evaluate.py first.")
    d = np.load(path, allow_pickle=True)
    y = d["y"]
    lg = {"efficientnet_b0": d["eff"][0], "resnet50": d["res"][0]}   # un-augmented view
    shipped = {n: C.load_temperature(n) for n in C.BACKBONES}

    folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED).split(y, y))
    crossfit_T = {n: [] for n in lg}
    cf_logits_T = {n: np.zeros(len(y)) for n in lg}   # per-image T used (from the other half)
    for fit_idx, eval_idx in folds:
        for n in lg:
            T = fit_temperature(lg[n][fit_idx], y[fit_idx])
            crossfit_T[n].append(T)
            cf_logits_T[n][eval_idx] = T

    probs = {}   # name -> (before, after_crossfit, shipped_insample)
    for n in lg:
        before = softmax(lg[n])
        after = softmax(lg[n] / cf_logits_T[n][:, None])
        ship = softmax(lg[n] / shipped[n])
        probs[n] = (before, after, ship)
    w = C.ENSEMBLE_WEIGHTS
    probs["ensemble"] = tuple(w["efficientnet_b0"] * probs["efficientnet_b0"][i]
                              + w["resnet50"] * probs["resnet50"][i] for i in range(3))

    out = {"split": "validation split, 2-fold cross-fitting", "seed": SEED, "n_bins": N_BINS,
           "n_images": int(len(y)),
           "shipped_temperatures": shipped,
           "crossfit_temperatures": {n: crossfit_T[n] for n in lg},
           "note": "'after_crossfit' fits T on one half and evaluates on the other half. "
                   "'shipped_insample' uses model/*_temp.pt, which were fitted on this same split.",
           "models": {}}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (name, (b, a, s)) in enumerate(probs.items()):
        out["models"][name] = {
            "ece_before": ece(b, y, N_BINS), "nll_before": nll(b, y),
            "ece_after_crossfit": ece(a, y, N_BINS), "nll_after_crossfit": nll(a, y),
            "ece_shipped_insample": ece(s, y, N_BINS), "nll_shipped_insample": nll(s, y),
            "accuracy_before": float((b.argmax(1) == y).mean()),
            "accuracy_after_crossfit": float((a.argmax(1) == y).mean()),
        }
        for row, (label, p) in enumerate([("before (T=1)", b), ("after (cross-fitted T)", a)]):
            ax = axes[row, col]
            xs, ys, ns = reliability_points(p, y)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(xs, ys, "o-", color="#4338CA")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
            ax.set_title(f"{name} - {label}\nECE {ece(p, y, N_BINS):.4f}")
    fig.suptitle("Reliability diagrams, validation split (15 bins)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "reliability_diagrams.png"), dpi=130)

    out["git"], out["date"] = C.git_info(), C.today()
    with open(os.path.join(C.RESULTS, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    for n, m in out["models"].items():
        print(f"{n:16s} ECE {m['ece_before']:.4f} -> {m['ece_after_crossfit']:.4f} (cross-fit)  "
              f"NLL {m['nll_before']:.4f} -> {m['nll_after_crossfit']:.4f}")


if __name__ == "__main__":
    main()
