"""
ui.py
Shared UI components for RetinaCare AI.
Purple theme, forced black text, and professional medical layout.
"""

import streamlit as st

def apply_medical_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Layout Reset */
        .block-container { padding-top: 1rem !important; }
        html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
        
        /* LIGHT PURPLE THEME */
        .stApp {
            background-color: #F3E8FF;
        }
        
        /* INK BLACK TEXT (Universal) */
        h1, h2, h3, h4, h5, p, span, li, label, div, .stMarkdown {
            color: #000000 !important;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 20px;
            padding: 2rem;
            border: 2px solid #A855F7;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        
        /* Prediction Badge */
        .prediction-badge {
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            font-weight: 800;
            font-size: 2.5rem;
            border: 5px solid;
            background: white;
            margin-bottom: 1rem;
        }

        /* Fixed Alert Text Visibility */
        div[data-testid="stNotification"] p, 
        div[data-testid="stAlert"] p, 
        div[data-testid="stExpander"] p {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def footer():
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #000000 !important; font-weight: 700;'>© 2026 RetinaCare AI Systems | Medical Research Tool</p>", unsafe_allow_html=True)
