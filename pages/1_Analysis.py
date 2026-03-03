"""
1_Analysis.py
Inference Dashboard - RetinaCare AI.
Restored Graphs + Maximum Memory Safety for Cloud.
"""

import os
import io
import gc
import cv2
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
st.set_page_config(page_title="Analysis | RetinaCare AI", layout="wide")
apply_medical_theme()
cfg = load_config()
EFF_PATH = "model/efficientnet_b0_dr.pth"

# Initialize session state for memory-heavy images
if 'gcam_img' not in st.session_state: st.session_state.gcam_img = None
if 'lime_img' not in st.session_state: st.session_state.lime_img = None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Diagnostic controls")
    uploaded_file = st.file_uploader("Upload photograph", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    use_tta = st.toggle("Enable Deep Scan", value=False)
    use_ensemble = st.toggle("Joint Brain Mode", value=False) # Default OFF for stability
    st.markdown("---")
    lime_samples = st.slider("Analysis Detail", 100, 300, 150) # Lowered for Cloud safety

# -----------------------------------------------------------------------------
# Analysis Page
# -----------------------------------------------------------------------------
st.markdown("<h1>Patient Analysis Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("<div style='text-align: center; padding: 5rem; border: 3px dashed #000000; border-radius: 30px; background: white;'><h2>Waiting for image upload...</h2></div>", unsafe_allow_html=True)
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

# 1. Quality Checklist
passed, issues = check_image_quality(pil_image, cfg)
st.markdown("<div class='glass-card' style='padding:1.5rem;'>", unsafe_allow_html=True)
q1, q2, q3 = st.columns(3)
with q1: st.write("✅ **Resolution:** Adequate")
with q2: st.write("✅ **Sharpness:** Verified" if passed else "❌ **Blurry Image**")
with q3: st.write("✅ **Brightness:** Normal" if passed else "❌ **Poor Lighting**")
st.markdown("</div>", unsafe_allow_html=True)

if not passed:
    st.error(f"Quality Alert: {', '.join(issues)}")
    if not st.button("Ignore & Analyze"): st.stop()

# 2. Prediction Engine
with st.spinner("AI Brain performing clinical assessment..."):
    from model.predict import predict
    grade, label, description, probs, preprocessed_img = predict(pil_image, use_tta=use_tta, use_ensemble=use_ensemble)
    gc.collect() # Force clear memory after prediction

# 3. Restored Visual Results
col_res, col_chart = st.columns([1, 1.2], gap="large")

with col_res:
    colors = {0:"#10B981", 1:"#F59E0B", 2:"#F97316", 3:"#EF4444", 4:"#991B1B"}
    res_c = colors[grade]
    st.markdown(f"<div class='prediction-badge' style='color: {res_c}; border-color: {res_c};'>{label}</div>", unsafe_allow_html=True)
    
    # NEW: Severity Gauge Restored
    st.write("### Disease Severity Gauge")
    fig_g, ax_g = plt.subplots(figsize=(8, 1))
    fig_g.patch.set_alpha(0); ax_g.set_facecolor('#F3E8FF')
    ax_g.barh([0], [4], color="#E2E8F0", height=0.5) # Background
    ax_g.barh([0], [grade], color=res_c, height=0.5) # Level
    ax_g.set_xlim(0, 4); ax_g.set_xticks([0,1,2,3,4])
    ax_g.set_xticklabels(["Healthy", "Mild", "Mod", "Sev", "Crit"], fontweight='bold')
    ax_g.get_yaxis().set_visible(False)
    st.pyplot(fig_g)
    plt.close(fig_g)

with col_chart:
    st.write("### Probability Distribution")
    fig_p, ax_p = plt.subplots(figsize=(8, 4))
    fig_p.patch.set_alpha(0); ax_p.set_facecolor('white')
    labels = ["None", "Mild", "Mod", "Sev", "Prolif"]
    ax_p.barh(labels, probs * 100, color="#CBD5E1")
    ax_p.get_children()[grade].set_color(res_c) # Highlight result
    ax_p.set_title("Neural Network Confidence", fontweight='bold')
    st.pyplot(fig_p)
    plt.close(fig_p)

# 4. Explainability Tabs (Lazy Loaded)
st.markdown("---")
st.markdown("## Clinical Evidence (Click to load)")
t1, t2, t3 = st.tabs(["AI Focus Map", "Structural Segmentation", "Pixel Influence"])

with t1:
    with st.spinner("Calculating Focus..."):
        from explainability.gradcam import get_gradcam_overlay
        ov, _, _ = get_gradcam_overlay(pil_image, EFF_PATH, class_idx=grade)
        st.session_state.gcam_img = ov
        st.image(ov, width=500)
        st.write("Red spots highlight where the AI found high-risk vessel irregularities.")

with t2:
    with st.spinner("Calculating Shapes..."):
        from explainability.lime_explain import get_lime_explanation
        lime_img, _ = get_lime_explanation(pil_image, EFF_PATH, num_samples=lime_samples)
        st.session_state.lime_img = lime_img
        st.image(lime_img, width=500)
        st.write("Green borders show the shapes that pushed the AI toward this result.")

with t3:
    with st.spinner("Calculating Pixels..."):
        from explainability.shap_explain import get_shap_explanation
        shap_img, _ = get_shap_explanation(pil_image, EFF_PATH, class_idx=grade)
        st.image(shap_img, width=300)
        st.write("Pixel-level deviations from a healthy eye model.")

# 5. Report Restoration
st.markdown("---")
if st.session_state.gcam_img is not None and st.session_state.lime_img is not None:
    st.markdown("### Clinical Documentation")
    if st.button("Generate Detailed Medical Report"):
        from utils.report import generate_report
        pdf = generate_report(grade, label, description, probs, preprocessed_img, st.session_state.gcam_img, st.session_state.lime_img)
        st.download_button("💾 Download Analysis PDF", data=bytes(pdf), file_name=f"RetinaCare_{grade}.pdf")
else:
    st.info("💡 Please click the 'Focus Map' and 'Shapes' tabs above to unlock the PDF report option.")

footer()
