"""
2_Evaluation.py
Evaluation dashboard. Shows ONLY numbers that were measured by the scripts in
scripts/ and saved under results/. If those files are missing no numbers are shown.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import auc, roc_curve

from utils.ui import apply_medical_theme, footer

st.set_page_config(page_title="Evaluation | DR Grader", layout="wide")
apply_medical_theme()

RESULTS = "results"
LABELS = ["None", "Mild", "Mod", "Sev", "Prolif"]


def _path(name: str) -> str:
    return os.path.join(RESULTS, name)


st.markdown("<h1>Model Evaluation</h1>", unsafe_allow_html=True)

if not os.path.exists(_path("val_metrics.json")) or not os.path.exists(_path("val_predictions.csv")):
    st.error("Results not found. Run `python scripts/evaluate.py` to generate them.")
    footer()
    st.stop()

with open(_path("val_metrics.json")) as f:
    metrics = json.load(f)
preds = pd.read_csv(_path("val_predictions.csv"))
variants = metrics["variants"]

st.info(
    "These numbers are measured on the **validation split** (`data/_val_split.csv`, "
    f"n = {next(iter(variants.values()))['n_images']}). The same split was used for early stopping and for "
    "fitting the temperature, so this is not a held-out test set. Dataset: APTOS 2019 only, 160 px images. "
    "Not a medical device."
)

st.markdown("### All model variants")
table = pd.DataFrame([{
    "Variant": name,
    "QWK": round(v["qwk"], 4),
    "95% CI (bootstrap)": f"{v['qwk_ci95'][0]:.4f} - {v['qwk_ci95'][1]:.4f}",
    "Accuracy": f"{v['accuracy'] * 100:.1f}%",
} for name, v in variants.items()])
st.dataframe(table, hide_index=True, use_container_width=True)
st.caption(f"Commit {metrics['git']['commit'][:8]}, measured {metrics['date']}. "
           f"Ensemble weights {metrics['ensemble_weights']}.")

names = list(variants)
choice = st.selectbox("Variant to inspect", names, index=names.index(metrics["primary_model"]))
v = variants[choice]

c1, c2, c3 = st.columns(3)
c1.metric("QWK", f"{v['qwk']:.4f}")
c2.metric("Accuracy", f"{v['accuracy'] * 100:.1f}%")
c3.metric("Validation images", v["n_images"])

col_cm, col_tab = st.columns([1.2, 1])
with col_cm:
    st.subheader("Confusion matrix (row-normalised)")
    cm = np.array(v["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples", xticklabels=LABELS,
                yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)
    plt.close(fig)
with col_tab:
    st.subheader("Per-grade metrics")
    st.table(pd.DataFrame(v["per_class"]).T.rename(index=lambda k: LABELS[int(k)]).round(3))

st.markdown("---")
st.subheader("One-vs-rest ROC curves")
sub = preds[preds.model == choice]
fig_roc, ax_roc = plt.subplots(figsize=(8, 4))
for i in range(5):
    fpr, tpr, _ = roc_curve(sub.y_true == i, sub[f"p{i}"])
    ax_roc.plot(fpr, tpr, label=f"{LABELS[i]} (AUC = {auc(fpr, tpr):.2f})")
ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
ax_roc.legend()
st.pyplot(fig_roc)
plt.close(fig_roc)

st.markdown("---")
st.subheader("Calibration")
if os.path.exists(_path("calibration.json")):
    with open(_path("calibration.json")) as f:
        cal = json.load(f)
    rows = [{"Model": n,
             "ECE before": round(m["ece_before"], 4),
             "ECE after (cross-fitted T)": round(m["ece_after_crossfit"], 4),
             "NLL before": round(m["nll_before"], 4),
             "NLL after (cross-fitted T)": round(m["nll_after_crossfit"], 4)}
            for n, m in cal["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Temperature fitted on one half of the validation split and evaluated on the other half "
               "(2-fold cross-fitting), 15 ECE bins.")
    if os.path.exists(_path("reliability_diagrams.png")):
        st.image(_path("reliability_diagrams.png"))
else:
    st.write("Not yet measured. Run `python scripts/calibration_report.py`.")

st.markdown("---")
st.subheader("Robustness to image perturbations")
if os.path.exists(_path("robustness.csv")):
    rob = pd.read_csv(_path("robustness.csv"))
    st.dataframe(rob.round(4), hide_index=True, use_container_width=True)
    if os.path.exists(_path("robustness.png")):
        st.image(_path("robustness.png"))
else:
    st.write("Not yet measured. Run `python scripts/robustness.py`.")

if os.path.exists(_path("ERROR_ANALYSIS.md")):
    st.markdown("---")
    with open(_path("ERROR_ANALYSIS.md"), encoding="utf-8") as f:
        st.markdown(f.read())

footer()
