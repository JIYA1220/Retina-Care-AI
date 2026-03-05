import streamlit as st
from PIL import Image
import numpy as np
from model.predict import predict as predict_fn
from explainability.gradcam import get_gradcam_overlay as gradcam_fn
from explainability.lime_explain import get_lime_explanation as lime_fn
from explainability.shap_explain import get_shap_explanation as shap_fn

@st.cache_data(show_spinner=False)
def get_cached_prediction(img, use_tta, use_ensemble):
    return predict_fn(img, use_tta=use_tta, use_ensemble=use_ensemble)

@st.cache_data(show_spinner=False)
def get_cached_gradcam(img, weights_path, class_idx):
    return gradcam_fn(img, weights_path=weights_path, class_idx=class_idx)

@st.cache_data(show_spinner=False)
def get_cached_lime(img, weights_path, num_samples):
    return lime_fn(img, weights_path=weights_path, num_samples=num_samples)

@st.cache_data(show_spinner=False)
def get_cached_shap(img, weights_path, class_idx):
    return shap_fn(img, weights_path=weights_path, class_idx=class_idx)
