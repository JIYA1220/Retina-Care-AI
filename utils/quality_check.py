"""
quality_check.py
Automated image quality validation for retinal fundus images.
Checks for blur, exposure, and resolution.
"""

import cv2
import numpy as np
from PIL import Image

def check_image_quality(pil_image, cfg):
    """
    Validates the quality of an uploaded retinal image.
    
    Returns:
        passed (bool): True if quality is acceptable.
        issues (list): List of strings describing found issues.
    """
    issues = []
    
    # Convert to grayscale for analysis
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. Resolution Check
    h, w = gray.shape
    if h < cfg.quality.min_resolution or w < cfg.quality.min_resolution:
        issues.append(f"Low resolution: {w}x{h} (Min recommended: {cfg.quality.min_resolution}px)")

    # 2. Blur Detection (Laplacian Variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < cfg.quality.blur_threshold:
        issues.append(f"Image is too blurry (Score: {blur_score:.1f}, Min: {cfg.quality.blur_threshold})")

    # 3. Brightness/Exposure Check
    mean_brightness = gray.mean()
    if mean_brightness < cfg.quality.min_brightness:
        issues.append(f"Image is too dark (Mean: {mean_brightness:.1f})")
    elif mean_brightness > cfg.quality.max_brightness:
        issues.append(f"Image is overexposed (Mean: {mean_brightness:.1f})")

    passed = len(issues) == 0
    return passed, issues
