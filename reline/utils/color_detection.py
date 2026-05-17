from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

ColorDetectMode = Literal['auto', 'force_color', 'force_gray']


@dataclass(frozen=True)
class ColorDetectionResult:
    is_color: bool
    saturated_ratio: Optional[float] = None
    rgb_diff_mean: Optional[float] = None
    reason: str = 'auto'


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        result = img.copy()
    else:
        result = np.asarray(img)
        if result.size and float(np.nanmax(result)) <= 1.0:
            result = result * 255.0
        result = np.clip(result, 0, 255).astype(np.uint8)

    if result.ndim == 2:
        return np.stack([result, result, result], axis=-1)
    if result.ndim == 3 and result.shape[2] >= 3:
        return result[:, :, :3]
    return np.squeeze(result)


def detect_image_color(img: np.ndarray, mode: ColorDetectMode | None = 'auto') -> ColorDetectionResult:
    mode = mode or 'auto'
    if mode == 'force_color':
        return ColorDetectionResult(True, reason='force_color')
    if mode == 'force_gray':
        return ColorDetectionResult(False, reason='force_gray')

    squeezed = np.squeeze(img)
    if squeezed.ndim == 2:
        return ColorDetectionResult(False, reason='single_channel')
    if squeezed.ndim != 3 or squeezed.shape[2] == 1:
        return ColorDetectionResult(False, reason='single_channel')

    rgb = _to_uint8_rgb(squeezed)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return ColorDetectionResult(False, reason='single_channel')

    height, width = rgb.shape[:2]
    scale = 256 / max(height, width)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)

    value = rgb.max(axis=2)
    mask = (value >= 10) & (value <= 245)
    if not np.any(mask):
        return ColorDetectionResult(False, 0.0, 0.0, 'no_sample_pixels')

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    rgb_diff = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)

    sampled_saturation = saturation[mask]
    sampled_diff = rgb_diff[mask]
    saturated_ratio = float(np.mean(sampled_saturation > 30))
    rgb_diff_mean = float(np.mean(sampled_diff))
    is_color = saturated_ratio > 0.05 and rgb_diff_mean > 10
    return ColorDetectionResult(is_color, saturated_ratio, rgb_diff_mean, 'auto')
