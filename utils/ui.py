"""
ui.py
Shared UI components and CSS for the Diabetic Retinopathy Severity Grader.
Provides a vibrant, emoji-free, high-contrast professional medical theme.
"""

import streamlit as st

def apply_medical_theme():
    """
    Applies the global professional medical theme with glassmorphism and animated backgrounds.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Global Reset */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        
        /* VERY COLORFUL ANIMATED GRADIENT BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #EEF2FF, #F0FDFA, #FAF5FF, #FFF7ED);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Professional Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
            margin-bottom: 2rem;
        }
        
        /* High-Contrast Bold Black Text */
        h1, h2, h3, h4, h5, p, span, li, label, div {
            color: #000000 !important;
        }
        
        /* Force black text inside Streamlit notification components */
        div[data-testid="stNotification"] p, 
        div[data-testid="stAlert"] p, 
        div[data-testid="stExpander"] p,
        .stAlert p {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Entrance Animation */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .stMarkdown, .stImage, .stButton { animation: fadeIn 0.8s ease-out; }
        
        /* Sidebar Styles */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-right: 1px solid #E2E8F0;
        }
        
        /* Prediction Badge Pulse */
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
        .prediction-badge {
            animation: pulse 2s infinite ease-in-out;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            font-weight: 800;
            font-size: 2.5rem;
            border: 4px solid;
        }
        </style>
    """, unsafe_allow_html=True)

def footer():
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748B !important; font-size: 12px;'>Built with PyTorch and Streamlit | APTOS 2019 Dataset Repository</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #EF4444 !important; font-size: 10px; font-weight: 700;'>WARNING: NOT FOR CLINICAL USE. This is a research evaluation aid only.</p>", unsafe_allow_html=True)
