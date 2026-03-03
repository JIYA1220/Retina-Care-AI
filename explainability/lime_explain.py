"""
lime_explain.py
LIME superpixel explanations for the DR grader.
Works fully on CPU - LIME does not need a GPU.
"""

import os
import numpy as np
import torch
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.model import load_model
from utils.preprocess import pil_to_preprocessed


VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def _batch_predict(images: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """
    LIME calls this with a batch of perturbed images (N,H,W,C) uint8.
    Returns softmax probabilities (N,5).
    """
    tensors = []
    for img in images:
        t = VAL_TRANSFORM(image=img.astype(np.uint8))["image"]
        tensors.append(t)
    batch = torch.stack(tensors)          # (N,3,H,W)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def get_lime_explanation(pil_image: Image.Image,
                         weights_path: str = "model/efficientnet_b0_dr.pth",
                         model_name: str = "efficientnet_b0",
                         num_samples: int = 300,
                         num_features: int = 10):
    """
    Returns
    -------
    lime_overlay : np.ndarray (H,W,3) uint8
    explanation  : lime_image.ImageExplanation
    """
    device = torch.device("cpu")
    
    # Demo Mode handling
    if not os.path.exists(weights_path):
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        return dummy, None

    model = load_model(weights_path, model_name, device)

    img_np = pil_to_preprocessed(pil_image, img_size=160)   # uint8 RGB

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_np,
        classifier_fn=lambda imgs: _batch_predict(imgs, model),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )

    lime_overlay = (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
    return lime_overlay, explanation
