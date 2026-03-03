"""
1_Analysis.py
Core diagnostic engine for Diabetic Retinopathy.
Optimized with Lazy Loading to prevent Out-of-Memory crashes on Cloud.
"""

import os
import io
import gc
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.quality_check import check_image_quality
from utils.ui import apply_medical_theme, footer

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analysis | DR Grader", layout="wide")
apply_medical_theme()
cfg = load_config()
EFF_PATH = "model/efficientnet_b0_dr.pth"

# -----------------------------------------------------------------------------
# Analysis Logic
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Diagnostic Controls")
    uploaded_file = st.file_uploader("Upload fundus photograph", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    use_tta = st.toggle("Enable Deep Scan", value=False) # Default to False for speed
    use_ensemble = st.toggle("Joint Brain Mode", value=True)
    lime_samples = st.slider("LIME Detail", 100, 500, 200) # Lowered for stability

st.markdown("<h1>Patient Analysis Pipeline</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("<div style='text-align: center; padding: 5rem; border: 3px dashed #000000; border-radius: 30px; background: white;'><h2>Please upload a photo to begin</h2></div>", unsafe_allow_html=True)
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

# Quality Check
passed, issues = check_image_quality(pil_image, cfg)
if not passed:
    st.warning(f"Quality Note: {', '.join(issues)}")

# 1. Prediction Core
with st.spinner("AI Brain performing inference..."):
    from model.predict import predict
    grade, label, description, probs, preprocessed_img = predict(pil_image, use_tta=use_tta, use_ensemble=use_ensemble)
    gc.collect() # Clear memory

# 2. Results Display
col_res, col_chart = st.columns([1, 1.2], gap="large")
with col_res:
    colors = {0:"#10B981", 1:"#F59E0B", 2:"#F97316", 3:"#EF4444", 4:"#991B1B"}
    res_c = colors[grade]
    st.markdown(f"<div class='prediction-badge' style='color: {res_c}; border-color: {res_c};'>Grade {grade} — {label}</div>", unsafe_allow_html=True)
    st.info(f"Summary: {description}")

with col_chart:
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_alpha(0); ax.set_facecolor('white')
    ax.barh(["None", "Mild", "Mod", "Sev", "Prolif"], probs * 100, color="#CBD5E1")
    ax.set_title("Probability Distribution")
    st.pyplot(fig)
    plt.close(fig)

# 3. Lazy-Loaded Explainability (CRITICAL: Only run when tab is clicked)
st.markdown("---")
st.markdown("## Clinical Evidence (Click a tab to load)")
t1, t2, t3 = st.tabs(["Neural Focus Map", "Shape Segmentation", "Pixel Analysis"])

# Placeholders for PDF
if 'gcam_img' not in st.session_state: st.session_state.gcam_img = None
if 'lime_img' not in st.session_state: st.session_state.lime_img = None

with t1:
    with st.spinner("Calculating Focus Map..."):
        from explainability.gradcam import get_gradcam_overlay
        ov, heat, _ = get_gradcam_overlay(pil_image, EFF_PATH, class_idx=grade)
        st.session_state.gcam_img = ov
        st.image(ov, width=500)
        st.write("Red spots highlight where the AI found high-risk vessel irregularities.")

with t2:
    with st.spinner("Calculating Shapes (this takes ~20s)..."):
        from explainability.lime_explain import get_lime_explanation
        lime_img, _ = get_lime_explanation(pil_image, EFF_PATH, num_samples=lime_samples)
        st.session_state.lime_img = lime_img
        st.image(lime_img, width=500)
        st.write("Green borders show the specific clusters that triggered the diagnosis.")

with t3:
    with st.spinner("Calculating Pixel Impact..."):
        from explainability.shap_explain import get_shap_explanation
        shap_img, _ = get_shap_explanation(pil_image, EFF_PATH, class_idx=grade)
        st.image(shap_img, width=400)
        st.write("Bright pixels represent significant deviations from a healthy eye.")

# 4. Reporting (Only enabled after XAI is loaded)
st.markdown("---")
if st.session_state.gcam_img is not None and st.session_state.lime_img is not None:
    if st.button("Generate and Download Medical Report"):
        from utils.report import generate_report
        pdf_bytes = generate_report(grade, label, description, probs, preprocessed_img, st.session_state.gcam_img, st.session_state.lime_img)
        st.download_button("💾 Save PDF Report", data=bytes(pdf_bytes), file_name=f"RetinaCare_{grade}.pdf")
else:
    st.info("💡 Please click the 'Focus Map' and 'Shapes' tabs above to unlock the PDF report.")

footer()
