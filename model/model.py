"""
model.py
Generalized model class for DR grading supporting different backbones.
Includes Streamlit caching for high-speed inference.
"""

import torch
import torch.nn as nn
import timm
import streamlit as st

NUM_CLASSES = 5

class DRGrader(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

def build_model(model_name: str = "efficientnet_b0", pretrained: bool = True) -> DRGrader:
    return DRGrader(model_name=model_name, pretrained=pretrained)

@st.cache_resource
def load_model(weights_path: str, model_name: str, device_str: str) -> DRGrader:
    """
    Cached model loader to prevent reloading from disk on every interaction.
    """
    device = torch.device(device_str)
    model = build_model(model_name=model_name, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
