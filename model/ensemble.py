"""
ensemble.py
Ensemble logic for EfficientNet-B0 and ResNet-50.
Uses soft-voting with calibrated probabilities.
"""

import os
import torch
import numpy as np
import streamlit as st
from model.model import get_model

class DRGraderEnsemble:
    def __init__(self, eff_path="model/efficientnet_b0_dr.pth", res_path="model/resnet50_dr.pth", device="cpu"):
        self.eff_model = get_model(eff_path, "efficientnet_b0", device)
        self.res_model = get_model(res_path, "resnet50", device)
        
        # Soft-voting weights (EfficientNet performed better at 0.80)
        self.w_eff = 0.6
        self.w_res = 0.4

    def _get_calibrated_probs(self, model, model_name, tensor):
        logits = model(tensor)
        
        # Look for model-specific temperature calibration
        temp_path = f"model/{model_name}_temp.pt"
        if os.path.exists(temp_path):
            T = torch.load(temp_path, map_location="cpu")
            logits = logits / T
            
        return torch.softmax(logits, dim=1)

    def predict_probs(self, tensor):
        """
        Returns calibrated weighted average of softmax probabilities.
        """
        with torch.no_grad():
            probs_eff = self._get_calibrated_probs(self.eff_model, "efficientnet_b0", tensor)
            probs_res = self._get_calibrated_probs(self.res_model, "resnet50", tensor)
            
            # Weighted average
            final_probs = (self.w_eff * probs_eff) + (self.w_res * probs_res)
            
        return final_probs.squeeze().cpu().numpy()

@st.cache_resource
def get_ensemble(eff_path: str, res_path: str, device: str = "cpu") -> DRGraderEnsemble:
    return DRGraderEnsemble(eff_path, res_path, device)
