"""
predict.py
Run inference on a single retinal image.
Supports ensemble mode, TTA, and a 'Mock Mode' for testing without weights.
"""

import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.model import load_model
from model.ensemble import DRGraderEnsemble
from utils.preprocess import pil_to_preprocessed


LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

CLINICAL_DESCRIPTIONS = {
    0: "Healthy retina. No microaneurysms or hemorrhages detected. Annual screening recommended.",
    1: "Early signs detected. Minor vessel swelling (microaneurysms) present. Monitoring advised.",
    2: "Clear clinical signs. Blocked blood vessels and leaks detected. Follow-up in 6 months.",
    3: "Advanced disease state. Significant vessel blockage and high risk of vision loss. Urgent referral.",
    4: "Critical proliferative state. Fragile new blood vessels forming. Immediate surgical consult required.",
}

VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def predict(pil_image: Image.Image,
            use_tta: bool = False,
            use_ensemble: bool = False):
    """
    Returns grade, label, description, probs, and preprocessed image.
    If weights are missing, returns confident mock data for UI testing.
    """
    device = torch.device("cpu")
    img_np = pil_to_preprocessed(pil_image, img_size=160)
    preprocessed_img = img_np.copy()

    eff_path = "model/efficientnet_b0_dr.pth"
    res_path = "model/resnet50_dr.pth"

    # --- MOCK MODE: If weights don't exist, return dummy data ---
    if not os.path.exists(eff_path):
        # Generate confident probabilities (one class > 75%)
        mock_probs = np.random.dirichlet(np.ones(5) * 0.1) 
        grade = int(np.argmax(mock_probs))
        # Ensure the winning class is actually high
        mock_probs[grade] = 0.75 + (np.random.random() * 0.2)
        mock_probs = mock_probs / mock_probs.sum()
        
        return grade, LABELS[grade], CLINICAL_DESCRIPTIONS[grade], mock_probs, preprocessed_img

    # --- REAL INFERENCE ---
    tta_transforms = [
        lambda x: x,
        lambda x: np.ascontiguousarray(x[:, ::-1]),
        lambda x: np.ascontiguousarray(x[::-1, :]),
        lambda x: np.rot90(x, k=1),
        lambda x: np.rot90(np.ascontiguousarray(x[:, ::-1]), k=1)
    ]

    all_probs = []
    iterations = tta_transforms if use_tta else [tta_transforms[0]]

    if use_ensemble and os.path.exists(res_path):
        from model.ensemble import get_ensemble
        engine = get_ensemble(eff_path=eff_path, res_path=res_path, device="cpu")
    else:
        from model.model import get_model
        model = get_model(eff_path, "efficientnet_b0", "cpu")

    with torch.no_grad():
        for transform in iterations:
            aug_img = transform(img_np)
            tensor = VAL_TRANSFORM(image=aug_img)["image"].unsqueeze(0).to(device)
            
            if use_ensemble and os.path.exists(res_path):
                probs = engine.predict_probs(tensor)
            else:
                logits = model(tensor)
                # Apply Model-Specific Temperature Scaling
                temp_path = "model/efficientnet_b0_temp.pt"
                if os.path.exists(temp_path):
                    T = torch.load(temp_path, map_location="cpu")
                    logits = logits / T
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
            
            all_probs.append(probs)

    final_probs = np.mean(all_probs, axis=0)
    grade = int(np.argmax(final_probs))
    
    return grade, LABELS[grade], CLINICAL_DESCRIPTIONS[grade], final_probs, preprocessed_img
