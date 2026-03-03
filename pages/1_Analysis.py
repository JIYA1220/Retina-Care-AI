"""
1_Analysis.py
Core diagnostic engine for Diabetic Retinopathy.
Fixed: SHAP sizing and Stable PDF Download Logic.
"""

import os
import io
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.quality_check import check_image_quality
from utils.ui import apply_medical_theme, footer

# -----------------------------------------------------------------------------
# Init & Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analysis | DR Grader", layout="wide")
apply_medical_theme()
cfg = load_config()
EFF_PATH = cfg.model.weights_path

# Force black text specifically for this page's boxes
st.markdown("""
    <style>
    div[data-testid="stNotification"] p { color: #000000 !important; font-weight: 600; }
    div[data-testid="stExpander"] p { color: #000000 !important; }
    .stAlert { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Explainability & Report Functions
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_gradcam(img, path, idx):
    from explainability.gradcam import get_gradcam_overlay
    return get_gradcam_overlay(img, path, class_idx=idx)

@st.cache_data
def get_cached_lime(img, path, samples):
    from explainability.lime_explain import get_lime_explanation
    return get_lime_explanation(img, path, num_samples=samples)

@st.cache_data
def get_cached_shap(img, path, idx):
    from explainability.shap_explain import get_shap_explanation
    return get_shap_explanation(img, path, class_idx=idx)

@st.cache_data
def get_cached_pdf(grade, label, desc, probs, orig, gcam, lime):
    from utils.report import generate_report
    return generate_report(grade, label, desc, probs, orig, gcam, lime)

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Diagnostic Controls")
    uploaded_file = st.file_uploader("Upload fundus photograph", type=["png", "jpg", "jpeg"])
    
    st.markdown("---")
    use_tta = st.toggle("Enable Test-Time Augmentation", value=cfg.tta.enabled)
    use_ensemble = st.toggle("Enable Joint AI (Ensemble)", value=True)
    
    st.markdown("---")
    lime_samples = st.slider("LIME Samples", 100, 1000, 300)
    show_preproc = st.checkbox("Show pre-processed image", value=True)
    
    st.markdown("---")
    st.write(f"**Backbone:** {cfg.model.backbone}")

# -----------------------------------------------------------------------------
# Analysis Page Logic
# -----------------------------------------------------------------------------
st.markdown("<h1>Patient Analysis Pipeline</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
        <div style='text-align: center; padding: 5rem; border: 3px dashed #000000; border-radius: 30px; background: white;'>
            <h2 style='color: #000000 !important;'>Ready for Input</h2>
            <p style='color: #000000 !important;'>Please upload a retinal fundus photograph in the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

# Quality Check
passed, issues = check_image_quality(pil_image, cfg)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### Image Quality Report")
cols_q = st.columns(4)
checks = [
    ("Resolution", "Adequate" if pil_image.size[0] >= 224 else "Low"),
    ("Sharpness", "Sharp" if passed else "Blurry"),
    ("Brightness", "Normal" if passed else "Check Exposure"),
    ("Circle", "Detected" if passed else "Review Crop")
]
for i, (label, val) in enumerate(checks):
    status = "✅" if "Adequate" in val or "Sharp" in val or "Normal" in val or "Detected" in val else "❌"
    cols_q[i].write(f"{status} **{label}:** {val}")
st.markdown("</div>", unsafe_allow_html=True)

if not passed:
    st.error(f"Quality Alert: {', '.join(issues)}")
    if not st.button("Proceed Anyway"): st.stop()

# Prediction Core
with st.spinner("AI Engine performing inference..."):
    from model.predict import predict
    grade, label, description, probs, preprocessed_img = predict(pil_image, use_tta=use_tta, use_ensemble=use_ensemble)

# Sections A: Preview
st.markdown("---")
col_orig, col_pre = st.columns(2)
with col_orig:
    st.markdown("#### Original Upload")
    st.image(pil_image, width=350)
    st.caption(f"Filename: {uploaded_file.name}")

with col_pre:
    if show_preproc:
        st.markdown("#### Ben Graham Pre-processing")
        st.image(preprocessed_img, width=350)

# Results
st.markdown("---")
col_res, col_chart = st.columns([1, 1.2], gap="large")

with col_res:
    colors = {0:"#10B981", 1:"#F59E0B", 2:"#F97316", 3:"#EF4444", 4:"#991B1B"}
    res_c = colors[grade]
    st.markdown(f"<div class='prediction-badge' style='color: {res_c}; border-color: {res_c};'>Grade {grade} — {label}</div>", unsafe_allow_html=True)
    st.info(f"**Clinical Summary:** {description}")

with col_chart:
    st.markdown("#### Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0); ax.set_facecolor('white')
    ax.barh(["None", "Mild", "Mod", "Sev", "Prolif"], probs * 100, color="#CBD5E1", height=0.6)
    ax.set_xlim(0, 105)
    st.pyplot(fig)

# Section E: Explainability
st.markdown("---")
st.markdown("## Interpretability Evidence")
t1, t2, t3 = st.tabs(["Grad-CAM Map", "LIME Superpixels", "SHAP Pixel Impact"])

# Calculate evidence
ov, heat, _ = get_cached_gradcam(pil_image, EFF_PATH, grade)
lime_img, _ = get_cached_lime(pil_image, EFF_PATH, lime_samples)
shap_img, _ = get_cached_shap(pil_image, EFF_PATH, grade)

with t1:
    c1, c2, c3 = st.columns(3)
    c1.image(preprocessed_img, caption="Input")
    c2.image(heat, caption="Raw Attention")
    c3.image(ov, caption="Clinical Overlay")

with t2:
    st.image(lime_img, width=350)
    st.write("Highlighted areas show specific clusters that drove the diagnosis.")

with t3:
    st.image(shap_img, width=300) # FURTHER REDUCED SIZE
    st.write("SHAP identifies pixel-level deviations from a healthy baseline eye.")

# Section G: Stable PDF Report
st.markdown("---")
st.markdown("### Clinical Documentation")
st.write("Download the formal assessment summary below.")

# Generate PDF in the background
pdf_bytes = get_cached_pdf(grade, label, description, probs, preprocessed_img, ov, lime_img)

st.download_button(
    label="⬇️ Download Clinical PDF Report",
    data=bytes(pdf_bytes),
    file_name=f"RetinaCare_Report_{grade}.pdf",
    mime="application/pdf",
    use_container_width=True
)

footer()
