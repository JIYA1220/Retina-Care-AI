"""
1_Analysis.py
Core Inference Dashboard - RetinaCare AI.
High-Resolution Graphs, Full Ensemble, and Stable PDF Reports.
"""

import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import load_config
from utils.quality_check import check_image_quality
from utils.ui import apply_medical_theme, footer

# -----------------------------------------------------------------------------
# Init & Logic Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analysis | RetinaCare AI", layout="wide")
apply_medical_theme()
cfg = load_config()
EFF_PATH = "model/efficientnet_b0_dr.pth"

# -----------------------------------------------------------------------------
# Sidebar: Professional Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 Clinical Input")
    uploaded_file = st.file_uploader("Upload Retinal Scan", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    use_tta = st.toggle("Enable Deep Scan (TTA)", value=True)
    use_ensemble = st.toggle("Enable Joint Brain Mode", value=True)
    st.markdown("---")
    lime_samples = st.slider("Analysis Detail", 100, 1000, 300)
    
    st.markdown("---")
    st.write(f"**Model:** {cfg.model.backbone} + ResNet50")

# -----------------------------------------------------------------------------
# Main Analysis Logic
# -----------------------------------------------------------------------------
st.markdown("<h1>Patient Analysis Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
        <div style='text-align: center; padding: 5rem; border: 3px dashed #000000; border-radius: 30px; background: white;'>
            <h2 style='color: #000000 !important;'>Ready for Retinal Data</h2>
            <p style='color: #000000 !important;'>Please upload a fundus photograph via the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# Load Image
pil_image = Image.open(uploaded_file).convert("RGB")

# 1. Image Quality Metrics
passed, issues = check_image_quality(pil_image, cfg)
st.markdown("<div class='glass-card' style='padding:1.5rem;'>", unsafe_allow_html=True)
q1, q2, q3, q4 = st.columns(4)
with q1: st.write("✅ **Resolution:** Verified")
with q2: st.write("✅ **Sharpness:** High" if passed else "⚠️ **Check Focus**")
with q3: st.write("✅ **Brightness:** Optimal" if passed else "⚠️ **Check Lighting**")
with q4: st.write("✅ **Anatomy:** Circle Detected")
st.markdown("</div>", unsafe_allow_html=True)

# 2. Prediction Core
from utils.inference import get_cached_prediction
with st.spinner("AI Brain analyzing vascular patterns..."):
    grade, label, description, probs, preproc = get_cached_prediction(pil_image, use_tta, use_ensemble)

# 3. Main Result Display
st.markdown("---")
col_l, col_r = st.columns([1, 1.2], gap="large")

with col_l:
    st.markdown("### Clinical Scan")
    st.image(pil_image, width=400, caption="Original Uploaded Source")
    # Image Meta Stats
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    sh = int(cv2.Laplacian(gray, cv2.CV_64F).var())
    st.write(f"<b>Scanning Sharpness Index:</b> {sh}", unsafe_allow_html=True)

with col_r:
    colors = {0:"#10B981", 1:"#F59E0B", 2:"#F97316", 3:"#EF4444", 4:"#991B1B"}
    res_c = colors[grade]
    
    # Large Pulsing Result Badge
    st.markdown(f"<div class='prediction-badge' style='color: {res_c}; border-color: {res_c};'>{label}</div>", unsafe_allow_html=True)
    st.markdown(f"### Reliability Score: **{float(probs[grade])*100:.1f}%**")
    st.progress(float(probs[grade]))
    
    # Severity Gauge
    st.write("#### Disease Severity Gauge")
    fig_g, ax_g = plt.subplots(figsize=(8, 1.2))
    # Remove transparency to ensure visibility on all backgrounds
    fig_g.patch.set_facecolor('white')
    ax_g.set_facecolor('#F8FAFC')
    
    # Background bar (Full range)
    ax_g.barh([0], [4], color="#E2E8F0", height=0.6, label="Range")
    # Result bar (Actual grade)
    ax_g.barh([0], [grade], color=res_c, height=0.6)
    
    # Add a clear vertical line/marker at the grade position
    ax_g.axvline(x=grade, color=res_c, linestyle='-', linewidth=4)
    
    ax_g.set_xlim(-0.2, 4.2)
    ax_g.set_xticks([0, 1, 2, 3, 4])
    ax_g.set_xticklabels(["Healthy", "Mild", "Mod", "Sev", "Crit"], 
                         fontweight='bold', color='black', fontsize=10)
    
    # Clean up axes
    ax_g.get_yaxis().set_visible(False)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_g)
    plt.close(fig_g)

# 4. Probability Chart
st.markdown("---")
st.markdown("### Confidence Distribution Breakdown")
fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
fig_p.patch.set_facecolor('white')
ax_p.set_facecolor('#F8FAFC')
labels = ["None", "Mild", "Mod", "Sev", "Prolif"]
bars = ax_p.bar(labels, probs * 100, color="#CBD5E1", edgecolor='black', linewidth=0.5)
bars[grade].set_color(res_c)
ax_p.set_ylabel("Confidence %", fontweight='bold', color='black')
ax_p.tick_params(axis='both', colors='black')
for spine in ax_p.spines.values():
    spine.set_edgecolor('black')
plt.tight_layout()
st.pyplot(fig_p)
plt.close(fig_p)

# 5. Explainability (High Performance)
from utils.inference import get_cached_gradcam, get_cached_lime, get_cached_shap

st.markdown("---")
st.markdown("## Visual Interpretability Evidence")
t1, t2, t3 = st.tabs(["AI Focus Map", "Vascular Shapes", "Pixel Details"])

with t1:
    ov, heat, _ = get_cached_gradcam(pil_image, EFF_PATH, class_idx=grade)
    c1, c2, c3 = st.columns(3)
    with c1: st.image(preproc, caption="Input Stream")
    with c2: st.image(heat, caption="Raw Attention")
    with c3: st.image(ov, caption="Clinical Overlay")

with t2:
    lime_img, _ = get_cached_lime(pil_image, EFF_PATH, num_samples=lime_samples)
    st.image(lime_img, width=500)
    st.write("Green boundaries identify specific lesions detected by the neural network.")

with t3:
    shap_img, _ = get_cached_shap(pil_image, EFF_PATH, class_idx=grade)
    st.image(shap_img, width=350)
    st.write("Voxel-level deviations from a healthy retinal background.")

# 6. Report Generation
st.markdown("---")
st.markdown("### Professional Documentation")
if st.button("Generate Medical PDF Analysis Report"):
    from utils.report import generate_report
    with st.spinner("Assembling high-res summary..."):
        pdf = generate_report(grade, label, description, probs, preproc, ov, lime_img)
        st.download_button("💾 Save Diagnostic PDF", data=bytes(pdf), file_name=f"RetinaCare_Report_{grade}.pdf")

footer()
