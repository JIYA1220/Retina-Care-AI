"""
evaluate.py
Deterministic evaluation of every model variant the app can use on the
validation split (data/_val_split.csv).

NOTE: this split was also used for early stopping (train.py) and for fitting the
temperature, so these are validation numbers, not held-out test numbers.

Outputs (results/):
  val_metrics.json, val_predictions.csv, val_logits.npz,
  confusion_matrix.png (primary model) + confusion_matrix_<variant>.png
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import _common as C
from utils.metrics import bootstrap_ci, qwk, softmax

LABELS = ["0 None", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]


def main():
    torch.manual_seed(0)
    df = C.load_val_df()
    y = df["diagnosis"].to_numpy()
    t0 = time.time()
    imgs = C.preprocess_val_images(df)
    print(f"preprocessed {len(imgs)} images in {time.time() - t0:.0f}s", flush=True)

    nets = {n: C.load_net(n) for n in C.BACKBONES}
    temps = {n: C.load_temperature(n) for n in C.BACKBONES}
    # logits[name] has shape (n_views, N, 5); view 0 is the un-augmented image
    logits = {}
    for n, net in nets.items():
        t0 = time.time()
        logits[n] = np.stack([C.logits_for(net, imgs, v) for v in C.TTA_VIEWS])
        print(f"{n}: logits for {len(C.TTA_VIEWS)} views in {time.time() - t0:.0f}s", flush=True)

    eff, res = "efficientnet_b0", "resnet50"
    te, tr = temps[eff], temps[res]
    probs = {
        "efficientnet_b0": softmax(logits[eff][0] / te),
        "resnet50": softmax(logits[res][0] / tr),
        "ensemble": C.ensemble_probs(logits[eff][0], logits[res][0], te, tr),
        "efficientnet_b0+tta": np.mean([softmax(l / te) for l in logits[eff]], axis=0),
        "ensemble+tta": np.mean([C.ensemble_probs(logits[eff][v], logits[res][v], te, tr)
                                 for v in range(len(C.TTA_VIEWS))], axis=0),
    }
    primary = C.cfg().model.backbone  # the model config.yaml points at

    ckpt = {n: {"file": os.path.relpath(C.checkpoint_path(n), C.ROOT).replace("\\", "/"),
                "sha256": C.sha256(C.checkpoint_path(n))} for n in C.BACKBONES}
    variants, rows = {}, []
    for name, p in probs.items():
        pred = p.argmax(1)
        lo, hi = bootstrap_ci(y, pred, qwk, 1000, seed=0)
        prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
        cm = confusion_matrix(y, pred, labels=range(5))
        used = C.BACKBONES if name.startswith("ensemble") else (name.split("+")[0],)
        variants[name] = {
            "checkpoints": [ckpt[n] for n in used],
            "tta": name.endswith("+tta"),
            "n_images": int(len(y)),
            "qwk": qwk(y, pred), "qwk_ci95": [lo, hi],
            "accuracy": float((pred == y).mean()),
            "per_class": {str(k): {"precision": float(prec[k]), "recall": float(rec[k]),
                                   "f1": float(f1[k]), "support": int(sup[k])} for k in range(5)},
            "confusion_matrix": cm.tolist(),
        }
        for i, id_ in enumerate(df["id_code"]):
            rows.append([id_, name, int(y[i]), int(pred[i]), *[float(v) for v in p[i]]])

        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=LABELS,
                    yticklabels=LABELS, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name} (validation split, n={len(y)})")
        fig.tight_layout()
        fig.savefig(os.path.join(C.RESULTS, f"confusion_matrix_{name.replace('+', '_')}.png"), dpi=130)
        if name == primary:
            fig.savefig(os.path.join(C.RESULTS, "confusion_matrix.png"), dpi=130)
        plt.close(fig)
        print(f"{name:22s} QWK {variants[name]['qwk']:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  acc {variants[name]['accuracy']:.4f}", flush=True)

    out = {
        "split": "data/_val_split.csv (validation split; also used for early stopping and temperature fitting)",
        "primary_model": primary,
        "ensemble_weights": C.ENSEMBLE_WEIGHTS,
        "temperatures": temps,
        "img_size": C.cfg().model.img_size,
        "bootstrap": {"resamples": 1000, "seed": 0, "level": 0.95},
        "git": C.git_info(), "date": C.today(), "environment": C.env_info(),
        "variants": variants,
    }
    with open(os.path.join(C.RESULTS, "val_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame(rows, columns=["id", "model", "y_true", "y_pred", *[f"p{k}" for k in range(5)]]) \
        .to_csv(os.path.join(C.RESULTS, "val_predictions.csv"), index=False)
    np.savez_compressed(os.path.join(C.RESULTS, "val_logits.npz"), y=y, ids=df["id_code"].to_numpy(),
                        eff=logits[eff], res=logits[res])
    v = variants[primary]
    print(f"\nPRIMARY ({primary}): QWK {v['qwk']:.4f} (95% CI {v['qwk_ci95'][0]:.4f}-{v['qwk_ci95'][1]:.4f}), "
          f"acc {v['accuracy']:.4f}, n={v['n_images']}")


if __name__ == "__main__":
    main()
