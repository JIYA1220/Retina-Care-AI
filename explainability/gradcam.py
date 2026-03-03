"""
gradcam.py
Gradient-weighted Class Activation Mapping (Grad-CAM)
for EfficientNet-B0 (timm backbone).

The target layer is the last convolutional block of the backbone.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.model import load_model
from utils.preprocess import pil_to_preprocessed


VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Returns a CAM heatmap normalised to [0,1], shape (H,W).
        """
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Pool gradients across spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()

        # Normalise
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def get_gradcam_overlay(pil_image: Image.Image,
                        weights_path: str = "model/efficientnet_b0_dr.pth",
                        model_name: str = "efficientnet_b0",
                        class_idx: int = None):
    """
    Returns
    -------
    overlay   : np.ndarray (H,W,3) uint8 - heatmap blended on original image
    heatmap   : np.ndarray (H,W,3) uint8 - pure heatmap
    cam_raw   : np.ndarray (H,W)   float - raw CAM values
    """
    device = torch.device("cpu")
    
    # Check if weights exist for real loading, otherwise return dummy for Demo Mode
    if not os.path.exists(weights_path):
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        return dummy, dummy, np.zeros((160, 160))

    model = load_model(weights_path, model_name, device)

    # Target last conv block of EfficientNet-B0
    target_layer = model.backbone.blocks[-1]
    gradcam = GradCAM(model, target_layer)

    img_np = pil_to_preprocessed(pil_image, img_size=160)  # uint8 RGB
    tensor = VAL_TRANSFORM(image=img_np)["image"].unsqueeze(0)

    cam = gradcam.generate(tensor, class_idx=class_idx)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (160, 160))

    # Colormap (OpenCV applyColorMap returns BGR)
    heatmap_bgr = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original (OpenCV addWeighted works in BGR)
    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(original_bgr, 0.5,
                                  heatmap_bgr,
                                  0.5, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb, heatmap_rgb, cam_resized
