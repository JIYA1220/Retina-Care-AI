"""
app.py
Landing Page - RetinaCare AI.
Introduction, Clinical Context, and Disease Scale.
"""

import streamlit as st
from utils.ui import apply_medical_theme, footer

st.set_page_config(page_title="RetinaCare AI | Home", layout="wide")
apply_medical_theme()

# Title
st.markdown("<h1 style='text-align: center; font-size: 4rem; font-weight: 900;'>RetinaCare AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.5rem; font-weight: 700; color: #4338CA !important;'>Advanced Neural Screening for Diabetic Retinopathy</p>", unsafe_allow_html=True)

# Step Guide
with st.expander("How the AI Pipeline Works", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("<div class='glass-card' style='text-align:center;'><b>1. Upload Image</b><br>Submit fundus photo</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='glass-card' style='text-align:center;'><b>2. Optimization</b><br>Ben Graham Method</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='glass-card' style='text-align:center;'><b>3. Diagnostics</b><br>Dual-Model Consensus</div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='glass-card' style='text-align:center;'><b>4. Validation</b><br>XAI Mapping + PDF</div>", unsafe_allow_html=True)

# Clinical Context Section
st.markdown("---")
st.markdown("## Understanding the Disease")
st.markdown("""
<div style='background: white; padding: 25px; border-radius: 20px; border: 2px solid #000000; font-size: 1.1rem;'>
<b>Diabetic Retinopathy (DR)</b> is a medical condition in which damage occurs to the retina due to diabetes. 
It is a leading cause of blindness. Early screening is the best way to prevent vision loss. 
The AI system looks for abnormal vessel growth, swelling, and fluid leakage (exudates) within the retinal tissues.
</div>
""", unsafe_allow_html=True)

# Severity Scale Restoration
st.markdown("### Severity Grading Scale")
s1, s2, s3, s4, s5 = st.columns(5)
grades = [
    ("Grade 0", "Healthy", "#10B981", "Clear retina, normal vessels."),
    ("Grade 1", "Mild", "#F59E0B", "Small vessel bulges detected."),
    ("Grade 2", "Moderate", "#F97316", "Significant vessel leaks found."),
    ("Grade 3", "Severe", "#EF4444", "Extensive vessel blockage."),
    ("Grade 4", "Critical", "#991B1B", "Advanced new vessel clusters.")
]
cols = [s1, s2, s3, s4, s5]
for i, (g, name, color, desc) in enumerate(grades):
    with cols[i]:
        st.markdown(f"""
            <div style='background: {color}; padding: 20px; border-radius: 15px; color: white !important; height: 160px; text-align: center;'>
                <h3 style='color: white !important;'>{g}</h3>
                <b style='color: white !important;'>{name}</b>
                <p style='color: white !important; font-size: 13px; margin-top: 10px;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

# System Status
st.markdown("---")
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### System Specifications")
    ca, cb = st.columns(2)
    with ca:
        st.write("**Architecture:** EfficientNet-B0 + ResNet-50 Ensemble")
        st.write("**Optimization:** Focal Loss + Calibrated Temperature Scaling")
    with cb:
        st.write("**Evaluation:** 0.8042 Quadratic Weighted Kappa (Excellent)")
        st.write("**Inference:** Local Host Optimized (No RAM caps)")
    st.markdown("</div>", unsafe_allow_html=True)

footer()
