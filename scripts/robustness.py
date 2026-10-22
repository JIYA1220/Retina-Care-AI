"""
robustness.py
No retraining. Perturb the validation images and report QWK per condition.

Perturbations are applied to the RAW fundus image (as a user's upload would be),
before the normal preprocessing pipeline. To keep Gaussian blur affordable the raw
image is first downscaled so its long side is WORK_SIDE pixels; the "clean" row is
computed through the same downscale so every condition is compared like for like.
(This is why the clean QWK here can differ slightly from scripts/evaluate.py.)

Severities:
  blur        Gaussian sigma = 0.5%, 1%, 2% of image width
  dark        brightness x0.7, x0.5, x0.3
  bright      brightness x1.3, x1.6, x2.0
  jpeg        quality 80, 50, 20
  noise       Gaussian noise sigma = 5, 15, 30 (on a 0-255 scale)

Models: EfficientNet-B0 (shipped temperature, no TTA) and the 0.6/0.4 ensemble.
Outputs: results/robustness.csv, results/robustness.png
"""

import io
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import _common as C
from utils.metrics import qwk
from utils.preprocess import pil_to_preprocessed

WORK_SIDE = 640
SEED = 0

CONDITIONS = (
    [("clean", "none", 0)]
    + [("blur", f"sigma={f * 100:g}% of width", s) for s, f in enumerate([0.005, 0.01, 0.02], 1)]
    + [("dark", f"x{f}", s) for s, f in enumerate([0.7, 0.5, 0.3], 1)]
    + [("bright", f"x{f}", s) for s, f in enumerate([1.3, 1.6, 2.0], 1)]
    + [("jpeg", f"quality={q}", s) for s, q in enumerate([80, 50, 20], 1)]
    + [("noise", f"sigma={q}", s) for s, q in enumerate([5, 15, 30], 1)]
)
PARAMS = {
    "blur": [0.005, 0.01, 0.02], "dark": [0.7, 0.5, 0.3], "bright": [1.3, 1.6, 2.0],
    "jpeg": [80, 50, 20], "noise": [5, 15, 30],
}


def perturb(img: np.ndarray, kind: str, severity: int, rng) -> np.ndarray:
    if kind == "clean":
        return img
    p = PARAMS[kind][severity - 1]
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=max(0.1, p * img.shape[1]))
    if kind in ("dark", "bright"):
        return np.clip(img.astype(np.float32) * p, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(p))
        return np.array(Image.open(buf).convert("RGB"))
    if kind == "noise":
        return np.clip(img.astype(np.float32) + rng.normal(0, p, img.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def load_raw(id_code: str) -> np.ndarray:
    bgr = cv2.imread(C.img_path(id_code))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = WORK_SIDE / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return rgb


def main():
    df = C.load_val_df()
    y = df["diagnosis"].to_numpy()
    size = C.cfg().model.img_size
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    batches = {c: [] for c in CONDITIONS}
    for k, id_code in enumerate(df["id_code"]):
        raw = load_raw(id_code)
        for cond in CONDITIONS:
            pert = perturb(raw, cond[0], cond[2], rng)
            batches[cond].append(pil_to_preprocessed(Image.fromarray(pert), img_size=size))
        if (k + 1) % 100 == 0:
            print(f"preprocessed {k + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)

    nets = {n: C.load_net(n) for n in C.BACKBONES}
    temps = {n: C.load_temperature(n) for n in C.BACKBONES}
    rows = []
    for cond in CONDITIONS:
        imgs = np.stack(batches[cond])
        le = C.logits_for(nets["efficientnet_b0"], imgs)
        lr = C.logits_for(nets["resnet50"], imgs)
        preds = {"efficientnet_b0": le.argmax(1),
                 "ensemble": C.ensemble_probs(le, lr, temps["efficientnet_b0"], temps["resnet50"]).argmax(1)}
        for model, pred in preds.items():
            rows.append({"model": model, "perturbation": cond[0], "severity": cond[2],
                         "parameter": cond[1], "qwk": qwk(y, pred),
                         "accuracy": float((pred == y).mean()), "n_images": len(y)})
        print(f"{cond[0]:7s} sev{cond[2]} {cond[1]:18s} " +
              "  ".join(f"{r['model']}: {r['qwk']:.4f}" for r in rows[-2:]), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RESULTS, "robustness.csv"), index=False)

    kinds = ["blur", "dark", "bright", "jpeg", "noise"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for ax, kind in zip(axes, kinds):
        for model, color in [("efficientnet_b0", "#4338CA"), ("ensemble", "#F59E0B")]:
            sub = res[(res.model == model) & (res.perturbation == kind)].sort_values("severity")
            clean = res[(res.model == model) & (res.perturbation == "clean")].qwk.iloc[0]
            ax.plot([0, *sub.severity], [clean, *sub.qwk], "o-", color=color, label=model)
        ax.set_title(kind); ax.set_xlabel("severity (1 = mild, 3 = strong)"); ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("QWK"); axes[0].legend(fontsize=8)
    fig.suptitle(f"QWK under perturbation, validation split (n={len(df)})")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "robustness.png"), dpi=130)


if __name__ == "__main__":
    main()
