"""
app.py
Home Page - RetinaCare AI.
Vibrant medical introduction and navigation hub.
"""

import streamlit as st
from utils.ui import apply_medical_theme, footer

st.set_page_config(page_title="RetinaCare AI | Home", layout="wide")
apply_medical_theme()

# Header
st.markdown("<h1 style='text-align: center; font-size: 4rem; font-weight: 900;'>RetinaCare AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.5rem; color: #2563EB !important; font-weight: 700;'>AI-Powered Retinal Analysis with Explainable Artificial Intelligence</p>", unsafe_allow_html=True)

# Visual Workflow
with st.expander("How it works", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("<div class='glass-card' style='text-align:center;'><b>1. Upload</b><br>Provide retinal photograph</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='glass-card' style='text-align:center;'><b>2. Scan</b><br>Neural Feature Extraction</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='glass-card' style='text-align:center;'><b>3. Verify</b><br>Dual-Brain Cross-Check</div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='glass-card' style='text-align:center;'><b>4. Report</b><br>Generate Clinical PDF</div>", unsafe_allow_html=True)

# Clinical Context
st.markdown("---")
st.markdown("## Clinical Context")
st.markdown("""
<div style='background: white; padding: 25px; border-radius: 20px; border: 1px solid #E2E8F0; color: black !important;'>
Diabetic Retinopathy is a leading cause of vision loss. Our AI system provides a standardized screening tool 
that identifies vascular irregularities like microaneurysms and hemorrhages. 
Early detection through AI can speed up treatment and prevent blindness.
</div>
""", unsafe_allow_html=True)

# Severity Scale Visual
st.markdown("### Severity Scale Reference")
s1, s2, s3, s4, s5 = st.columns(5)
severity_info = [
    ("Grade 0", "No Disease", "#10B981", "Clear retina."),
    ("Grade 1", "Mild", "#F59E0B", "Microaneurysms."),
    ("Grade 2", "Moderate", "#F97316", "Vessel leaks."),
    ("Grade 3", "Severe", "#EF4444", "Blockages."),
    ("Grade 4", "Proliferative", "#991B1B", "New growth.")
]
cols = [s1, s2, s3, s4, s5]
for i, (grade, name, color, desc) in enumerate(severity_info):
    with cols[i]:
        st.markdown(f"""
            <div style='background: {color}; padding: 20px; border-radius: 15px; color: white !important; height: 160px; text-align: center;'>
                <h3 style='color: white !important;'>{grade}</h3>
                <b style='color: white !important;'>{name}</b><br>
                <p style='color: white !important; font-size: 13px;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

# Specifications
st.markdown("---")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### System Specifications")
ca, cb = st.columns(2)
with ca:
    st.write("**Backbone Architecture:** EfficientNet-B0 + ResNet-50 Ensemble")
    st.write("**Calibration Method:** Temperature Scaling (Platt Scaling)")
with cb:
    st.write("**Evaluation Metric:** 0.8042 Quadratic Weighted Kappa")
    st.write("**XAI Engine:** Grad-CAM | LIME | SHAP")
st.markdown("</div>", unsafe_allow_html=True)

footer()
