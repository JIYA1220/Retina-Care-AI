"""
3_About.py
Documentation and technical specifications.
Project overview, architecture, and technology stack.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.ui import apply_medical_theme, footer

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
st.set_page_config(page_title="About | DR Grader", layout="wide")
apply_medical_theme()

# -----------------------------------------------------------------------------
# Page Layout
# -----------------------------------------------------------------------------
st.markdown("<h1>System Documentation</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Project Overview", "Technical Architecture", "Dataset Stats"])

with tab1:
    st.markdown("### Project Objective")
    st.write("""
    This project was developed to provide an end-to-end automated screening tool for Diabetic Retinopathy.
    The pipeline integrates state-of-the-art computer vision with multiple layers of explainability to ensure 
    that clinicians can trust the AI's findings.
    """)
    
    st.markdown("### Decision Flow")
    st.markdown("""
    Input Image → **Ben Graham Preprocessing** → **EfficientNet-B0 + ResNet-50** → **Temperature Scaling** → **Ensemble Voting** → **Grade Result**
    """)
    
    st.markdown("### Technology Stack")
    st.write("The system is built on a modern AI stack:")
    cols = st.columns(4)
    cols[0].write("**Deep Learning:** PyTorch, timm")
    cols[1].write("**Explainability:** LIME, SHAP, Grad-CAM")
    cols[2].write("**Processing:** OpenCV, Albumentations")
    cols[3].write("**Interface:** Streamlit, Matplotlib")

with tab2:
    st.markdown("### Model Architecture")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("**Primary Backbone:** EfficientNet-B0 (Compound Scaling)")
    st.write("**Secondary Backbone:** ResNet-50 (Residual Connections)")
    st.write("**Classifier Head:** Dropout(0.3) → Linear(256) → ReLU → Dropout(0.15) → Linear(5)")
    st.write("**Ensemble Strategy:** Soft-Voting (Weighted Average)")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Why SHAP and LIME?")
    st.write("""
    - **Grad-CAM:** Identifies 'where' the model looks using gradients.
    - **LIME:** Identifies 'what shapes' matter by perturbing the image.
    - **SHAP:** Provides 'theoretically fair' pixel-level attribution.
    """)

with tab3:
    st.markdown("### Dataset Statistics")
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
        st.write(f"Total training samples: **{len(df)}**")
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0); ax.set_facecolor('white')
        counts = df['diagnosis'].value_counts().sort_index()
        ax.bar(["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"], counts, color="#6366F1")
        ax.set_title("Class Distribution (APTOS 2019)")
        st.pyplot(fig)
    else:
        st.info("Train metadata not found. Please ensure data/train.csv is present.")

st.markdown("---")
st.markdown("### Disclaimer")
st.error("""
    This software is provided for research and educational purposes only. It is not intended 
    for primary diagnosis. All findings should be reviewed by a licensed medical professional.
""")

footer()
