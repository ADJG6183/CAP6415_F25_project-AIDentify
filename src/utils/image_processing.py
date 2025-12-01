"""
Image processing utilities for AI-generated image detection.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load and preprocess an image.

    Args:
        image_path: Path to the image file
        target_size: Optional target size (height, width) to resize the image

    Returns:
        Preprocessed image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    return img


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def extract_patches(image: np.ndarray, patch_size: int = 64, stride: int = 32) -> np.ndarray:
    """
    Extract overlapping patches from an image.

    Args:
        image: Input image
        patch_size: Size of each patch
        stride: Stride between patches

    Returns:
        Array of patches with shape (num_patches, patch_size, patch_size, channels)
    """
    h, w = image.shape[:2]
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return np.array(patches)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.

    Args:
        image: RGB image

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def calculate_image_quality_metrics(image: np.ndarray) -> dict:
    """
    Calculate various image quality metrics.

    Args:
        image: Input image

    Returns:
        Dictionary of quality metrics
    """
    gray = convert_to_grayscale(image)

    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # Calculate contrast
    contrast = gray.std()

    # Calculate brightness
    brightness = gray.mean()

    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'brightness': brightness
    }
