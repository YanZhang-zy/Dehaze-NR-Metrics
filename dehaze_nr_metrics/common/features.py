"""Contrast and color helpers shared by FADE and BIQME.

Adapted from the original UT Austin FADE release.
See THIRD_PARTY_NOTICES.md for attribution and the original copyright notice.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def matlab_rgb2gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """Match MATLAB rgb2gray on uint8 (not CE.m's rounded 0.299 formula)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    gray = (rgb[..., 0] * 0.298936021293775
            + rgb[..., 1] * 0.587043074451121
            + rgb[..., 2] * 0.114020904255103)
    return np.clip(np.floor(gray + 0.5), 0, 255).astype(np.uint8)


def _contrast_energy(rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    sigma = 3.25
    # MATLAB -9.75 : 1 : 9.75 ends at 9.25 (20, not 21, samples).
    x = np.arange(20, dtype=np.float64) - 9.75
    gauss = np.exp(x * x / (-2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)
    gauss /= gauss.sum()
    gx = (x * x / sigma ** 4 - 1 / sigma ** 2) * gauss
    gx -= gx.sum() / x.size
    gx /= (0.5 * x * x * gx).sum()
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    channels = (0.299 * r + 0.587 * g + 0.114 * b, 0.5 * r + 0.5 * g - b, r - g)
    thresholds = np.array([9.225496406318721e-4, 8.969246659629488e-4, 2.069284034165411e-4]) * 255
    result = []
    for channel, threshold in zip(channels, thresholds):
        # border_in.m duplicates existing strips without reversing them.
        padded = np.concatenate((channel[:10], channel, channel[-10:]), axis=0)
        padded = np.concatenate((padded[:, :10], padded, padded[:, -10:]), axis=1)
        # For an even kernel MATLAB conv2(...,'same') starts at full index
        # len(kernel)//2; ndimage convolution origin=0 uses that alignment.
        cx = ndimage.convolve1d(padded, gx, axis=1, mode="constant", cval=0, origin=0)
        cy = ndimage.convolve1d(padded, gx, axis=0, mode="constant", cval=0, origin=0)
        contrast = np.sqrt(cx * cx + cy * cy)[10:-10, 10:-10]
        maximum = contrast.max()
        with np.errstate(invalid="ignore", divide="ignore"):
            response = contrast * maximum / (contrast + maximum * 0.1) - threshold
        result.append(np.where(response > 1e-7, response, 0))
    return tuple(result)
