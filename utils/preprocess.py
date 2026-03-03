"""
preprocess.py
Ben Graham-style preprocessing for retinal fundus images.
Crops to the retinal circle, enhances contrast, resizes.
"""

import cv2
import numpy as np
from PIL import Image


def crop_to_circle(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """Crop black borders around the retinal circle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y: y + h, x: x + w]


def ben_graham_preprocess(img: np.ndarray, img_size: int = 224) -> np.ndarray:
    """
    Ben Graham preprocessing:
    1. Resize to target size keeping the retina centred
    2. Subtract a Gaussian-blurred version to enhance local features
    3. Clip and normalise to [0, 255]
    """
    img = crop_to_circle(img)
    img = cv2.resize(img, (img_size, img_size))

    # Subtract blurred image to highlight fine vessels / lesions
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=img_size // 30)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)

    # Create a circular mask so the black border stays black
    mask = np.zeros_like(enhanced)
    cx, cy = img_size // 2, img_size // 2
    radius = int(img_size * 0.45)
    cv2.circle(mask, (cx, cy), radius, (1, 1, 1), -1)
    enhanced = enhanced * mask + 128 * (1 - mask)

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def load_and_preprocess(image_path: str, img_size: int = 224) -> np.ndarray:
    """Load an image file and return a preprocessed numpy array (RGB)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = ben_graham_preprocess(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pil_to_preprocessed(pil_image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Accept a PIL image (from Streamlit uploader) and preprocess it."""
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed = ben_graham_preprocess(img_bgr, img_size)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return processed_rgb
