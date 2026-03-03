"""
app.py
Home Page for Diabetic Retinopathy Severity Grader.
Clinical context, severity scale, and forced black text for readability.
"""

import streamlit as st
from utils.ui import apply_medical_theme, footer

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RetinaCare AI | Home", layout="wide")

# Force Apply Theme with specific Alert Box fix for black text
apply_medical_theme()
st.markdown("""
    <style>
    /* Specific fix for Streamlit Alert/Info boxes to ensure black text */
    div[data-testid="stNotification"] p { color: #000000 !important; }
    div[data-testid="stExpander"] p { color: #000000 !important; }
    .stAlert { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-size: 4rem; font-weight: 900;'>Diabetic Retinopathy Severity Grader</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.5rem; font-weight: 700; color: #4338CA !important;'>AI-Powered Retinal Analysis with Explainable Artificial Intelligence</p>", unsafe_allow_html=True)

with st.expander("How it works", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("<div class='glass-card' style='text-align:center;'><b>1. Upload</b><br>Provide retinal fundus image</div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='glass-card' style='text-align:center;'><b>2. Process</b><br>Ben Graham Enhancement</div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='glass-card' style='text-align:center;'><b>3. Analysis</b><br>Deep Neural Inference</div>", unsafe_allow_html=True)
    with c4: st.markdown("<div class='glass-card' style='text-align:center;'><b>4. Report</b><br>XAI Evidence and PDF</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# About the Disease Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Clinical Context")
st.markdown("""
<div style='background: white; padding: 20px; border-radius: 15px; border: 1px solid #000000;'>
Diabetic Retinopathy is a leading cause of blindness worldwide, affecting millions of people with diabetes. 
It is caused by damage to the small blood vessels in the retina. Early detection is critical as the disease 
often has no symptoms until vision loss occurs.
</div>
""", unsafe_allow_html=True)

st.markdown("### Severity Scale Visual")
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
            <div style='background: {color}; padding: 20px; border-radius: 15px; color: white !important; height: 150px; text-align: center;'>
                <h3 style='color: white !important;'>{grade}</h3>
                <b style='color: white !important;'>{name}</b><br>
                <p style='color: white !important; font-size: 13px;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Dataset & Model Info Card
# -----------------------------------------------------------------------------
st.markdown("---")
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Dataset and Model Specifications")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Dataset Source:** APTOS 2019 Blindness Detection")
        st.write("**Training Images:** 3,662 Fundus Photographs")
    with col_b:
        st.write("**Architecture:** EfficientNet-B0 + ResNet-50 Ensemble")
        st.write("**Inference:** Optimized for CPU Performance")
    st.markdown("</div>", unsafe_allow_html=True)

footer()
