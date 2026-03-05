"""
shap_explain.py
SHAP (SHapley Additive exPlanations) for EfficientNet-B0.
Optimized for CPU and robust against indexing errors.
"""

import torch
import numpy as np
import shap
import os
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.model import load_model
from utils.preprocess import pil_to_preprocessed

VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def disable_inplace_activations(model):
    for name, module in model.named_modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

def get_shap_explanation(pil_image: Image.Image,
                         weights_path: str = "model/efficientnet_b0_dr.pth",
                         model_name: str = "efficientnet_b0",
                         class_idx: int = 0):
    device = torch.device("cpu")
    if not os.path.exists(weights_path):
        return np.zeros((160, 160, 3), dtype=np.uint8), None

    from model.model import get_model
    model = get_model(weights_path, model_name, "cpu")
    model.eval()
    disable_inplace_activations(model)

    # Lightweight background
    background = torch.zeros((5, 3, 160, 160)) 
    
    img_np = pil_to_preprocessed(pil_image, img_size=160)
    input_tensor = VAL_TRANSFORM(image=img_np)["image"].unsqueeze(0).to(device)

    # GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_tensor)
    
    # Robust Indexing: Handle list of arrays or single array
    if isinstance(shap_values, list):
        # List of 5 arrays (one per class)
        sv = shap_values[class_idx]
    else:
        # Single array of shape (batch, classes, ...) or similar
        if len(shap_values.shape) == 5: # (batch, classes, c, h, w)
            sv = shap_values[:, class_idx]
        else:
            sv = shap_values # Fallback
            
    # Remove batch dimension and aggregate
    if len(sv.shape) > 3:
        sv = sv[0]
        
    sv_agg = np.abs(sv).mean(axis=0)
    
    # Normalize and colorize
    sv_norm = (sv_agg - sv_agg.min()) / (sv_agg.max() - sv_agg.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * sv_norm), cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), sv_agg
