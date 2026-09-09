"""Spatial analysis functions for the VIIRS-based urban growth study."""

from __future__ import annotations

import cv2
import numpy as np


class ContourResult(list):
    """List-like result that also preserves a rendered contour image."""

    def __init__(self, contours, contour_image):
        super().__init__(contours)
        self.contours = list(contours)
        self.drawn_image = contour_image


def morphological_clean(binary_mask):
    """Clean a binary mask using morphological opening and closing."""
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[-1] in (3, 4):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY if mask.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported image shape for binary_mask")
    elif mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array or a color image")

    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def find_and_draw_contours(binary_mask):
    """Find contours of bright clusters and return a list-like result with the draw image."""
    mask = np.asarray(binary_mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        elif mask.shape[-1] in (3, 4):
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY if mask.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported image shape for binary_mask")
    elif mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array or a color image")

    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rendered = np.zeros_like(binary)
    rendered = cv2.drawContours(rendered, contours, -1, 255, 1)
    return ContourResult(contours, rendered)


def calculate_centroid(contour):
    """Calculate the centroid of a contour using spatial moments."""
    contour_array = np.asarray(contour, dtype=np.float32)
    if contour_array.size == 0:
        raise ValueError("contour must not be empty")

    moments = cv2.moments(contour_array)
    m00 = moments.get("m00", 0.0)
    if m00 == 0:
        return np.array([0.0, 0.0], dtype=float)

    centroid_x = moments["m10"] / m00
    centroid_y = moments["m01"] / m00
    return np.array([centroid_x, centroid_y], dtype=float)
