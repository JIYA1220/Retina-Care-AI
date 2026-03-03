"""
1_Analysis.py
Inference Dashboard - RetinaCare AI.
Memory-optimized for Streamlit Cloud stability.
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
st.set_page_config(page_title="Analysis | RetinaCare AI", layout="wide")
apply_medical_theme()
cfg = load_config()
EFF_PATH = "model/efficientnet_b0_dr.pth"

# -----------------------------------------------------------------------------
# Memory-Safe Explainability (Calculating only when requested)
# -----------------------------------------------------------------------------
if 'gcam_img' not in st.session_state: st.session_state.gcam_img = None
if 'lime_img' not in st.session_state: st.session_state.lime_img = None

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 Controls")
    uploaded_file = st.file_uploader("Upload Retinal Photograph", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    use_tta = st.toggle("Multi-Pass Precision", value=False)
    use_ensemble = st.toggle("Joint Brain Mode", value=True)
    st.markdown("---")
    lime_samples = st.slider("LIME Depth", 100, 500, 200)

# -----------------------------------------------------------------------------
# Main Layout
# -----------------------------------------------------------------------------
st.markdown("<h1>Patient Analysis Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("<div style='text-align: center; padding: 5rem; border: 3px dashed #CBD5E1; border-radius: 30px; background: white;'><h2>Awaiting Clinical Data</h2><p>Please upload a retinal scan via the sidebar to start.</p></div>", unsafe_allow_html=True)
    st.stop()

pil_image = Image.open(uploaded_file).convert("RGB")

# 1. Quality Check
passed, issues = check_image_quality(pil_image, cfg)
if not passed:
    st.warning(f"Quality Alert: {', '.join(issues)}")

# 2. Results Section
col_l, col_r = st.columns([1, 1.2], gap="large")

with col_l:
    st.markdown("### Clinical Scan")
    st.image(pil_image, width=350, caption="Original Upload")
    
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY) if 'cv2' in globals() else None
    if gray is not None:
        st.metric("Sharpness", int(cv2.Laplacian(gray, cv2.CV_64F).var()))

with col_r:
    with st.spinner("Analyzing physiological markers..."):
        from model.predict import predict
        grade, label, description, probs, preproc = predict(pil_image, use_tta=use_tta, use_ensemble=use_ensemble)
        gc.collect() # Immediate memory flush

    colors = {0:"#10B981", 1:"#F59E0B", 2:"#F97316", 3:"#EF4444", 4:"#991B1B"}
    res_c = colors[grade]
    
    st.markdown(f"""
        <div class='prediction-badge' style='color: {res_c}; border-color: {res_c};'>
            {label} (Grade {grade})
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"### Diagnostic Reliability: **{float(probs[grade])*100:.1f}%**")
    st.progress(float(probs[grade]))
    st.info(f"**Medical Summary:** {description}")

# 3. Explainability Tabs (Memory Safe)
st.markdown("---")
st.markdown("## Visual Evidence Mapping")
t1, t2, t3 = st.tabs(["AI Focus Map", "Structural Segmentation", "Pixel Analysis"])

with t1:
    with st.spinner("Calculating focus..."):
        from explainability.gradcam import get_gradcam_overlay
        ov, heat, _ = get_gradcam_overlay(pil_image, EFF_PATH, class_idx=grade)
        st.session_state.gcam_img = ov
        st.image(ov, width=450)
        st.write("Heatmap showing regions of high neural attention (red).")

with t2:
    with st.spinner("Calculating shapes..."):
        from explainability.lime_explain import get_lime_explanation
        lime_img, _ = get_lime_explanation(pil_image, EFF_PATH, num_samples=lime_samples)
        st.session_state.lime_img = lime_img
        st.image(lime_img, width=450)
        st.write("Outline of structures that pushed the AI toward this result.")

with t3:
    with st.spinner("Calculating pixels..."):
        from explainability.shap_explain import get_shap_explanation
        shap_img, _ = get_shap_explanation(pil_image, EFF_PATH, class_idx=grade)
        st.image(shap_img, width=350)
        st.write("Pixel-level deviations from a healthy eye model.")

# 4. Report Generation
st.markdown("---")
if st.session_state.gcam_img is not None and st.session_state.lime_img is not None:
    st.markdown("### Clinical Documentation")
    if st.button("Generate Detailed Analysis PDF"):
        from utils.report import generate_report
        with st.spinner("Assembling report..."):
            pdf = generate_report(grade, label, description, probs, preproc, st.session_state.gcam_img, st.session_state.lime_img)
            st.download_button("💾 Save Medical PDF", data=bytes(pdf), file_name=f"RetinaCare_Report_{grade}.pdf")
else:
    st.info("💡 Please click the 'Focus Map' and 'Shapes' tabs above to unlock the PDF download.")

footer()
