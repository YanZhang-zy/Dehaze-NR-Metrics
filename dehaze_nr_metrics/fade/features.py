"""FADE patch features and statistical model helpers.

Adapted from the original UT Austin FADE release.
See THIRD_PARTY_NOTICES.md for attribution and the original copyright notice.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


from ..common.models import resolve_models
MODEL_FILES = (
    "natural_fogfree_image_features_ps8.mat",
    "natural_foggy_image_features_ps8.mat",
)
FEATURE_NAMES = (
    "MSCN_var", "MSCN_V_pair_R_var", "MSCN_V_pair_L_var", "Mean_sigma",
    "Mean_cv", "Mean_CE_gray", "Mean_CE_by", "Mean_CE_rg", "IE",
    "Mean_Id", "Mean_Is", "CF",
)


def model_fingerprints(model_dir: str | Path | None = None) -> dict:
    directory = resolve_models("fade", model_dir)
    return {
        name: {"path": str((directory / name).resolve()),
               "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
        for name in MODEL_FILES
    }


@lru_cache(maxsize=8)
def _load_models(directory: str):
    models = []
    for filename, suffix in zip(MODEL_FILES, ("fogfree", "foggy")):
        values = loadmat(Path(directory) / filename)
        mean = np.asarray(values[f"mu_{suffix}param"], dtype=np.float64).reshape(12)
        covariance = np.asarray(values[f"cov_{suffix}param"], dtype=np.float64)
        if covariance.shape != (12, 12) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError(f"Invalid FADE model: {filename}")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        models.append((mean, covariance))
    return tuple(models)


def load_models(model_dir: str | Path | None = None):
    """Read the original pair of 12-feature Gaussian models (cached read-only)."""
    directory = (resolve_models("fade", model_dir)).resolve()
    return _load_models(str(directory))


def _rgb_uint8(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            if opened.mode not in {"1", "L", "LA", "P", "PA", "RGB", "RGBA"}:
                raise TypeError("FADE requires an explicit conversion for non-8-bit image files")
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        if image.mode not in {"1", "L", "LA", "P", "PA", "RGB", "RGBA"}:
            raise TypeError("FADE requires an explicit conversion for non-8-bit PIL images")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image)
        if rgb.dtype != np.uint8:
            raise TypeError("FADE expects uint8 RGB; floating point images must be converted explicitly")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("FADE expects an H x W x 3 RGB image")
    if min(rgb.shape[:2]) < 16:
        raise ValueError("Original FADE CE border copying needs cropped height and width >= 16")
    return rgb[: rgb.shape[0] // 8 * 8, : rgb.shape[1] // 8 * 8, :3]


def _patches(array: np.ndarray) -> np.ndarray:
    """Rows are MATLAB im2col distinct columns, including column-major order."""
    rows, cols = array.shape
    return array.reshape(rows // 8, 8, cols // 8, 8).transpose(2, 0, 3, 1).reshape(-1, 64)


def _nanvar_rows(array: np.ndarray) -> np.ndarray:
    # MATLAB var(x,0) uses denominator max(non-NaN count - 1, 1), so a
    # singleton is zero and an empty column is NaN. np.nanvar(ddof=1) alone
    # differs on singleton columns, which do occur in the paired features.
    present = ~np.isnan(array)
    count = present.sum(axis=1)
    total = np.where(present, array, 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
        dev = array - mean[:, None]
        result = np.where(present, dev * dev, 0).sum(axis=1) / np.maximum(count - 1, 1)
    result[count == 0] = np.nan
    return result


def _patch_entropy(gray: np.ndarray) -> np.ndarray:
    patches = _patches(gray).astype(np.int64)
    # One 256-bin histogram per patch, exactly entropy(uint8(column)).
    offsets = np.arange(patches.shape[0], dtype=np.int64)[:, None] * 256
    counts = np.bincount((patches + offsets).ravel(), minlength=patches.shape[0] * 256).reshape(-1, 256)
    probabilities = counts / 64.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(counts > 0, probabilities * np.log2(probabilities), 0)
    return -terms.sum(axis=1)


def _distances(features: np.ndarray, model, patch_variance: np.ndarray) -> np.ndarray:
    mean, covariance = model
    difference = mean[None, :] - features
    # Keep FADE.m's scalar-to-every-entry broadcast, not scalar * identity.
    matrices = (covariance[None, :, :] + patch_variance[:, None, None]) / 2
    finite = np.isfinite(difference).all(axis=1) & np.isfinite(matrices).all(axis=(1, 2))
    result = np.full(len(features), np.nan, dtype=np.float64)
    if finite.any():
        # Row right division b/A is solve(A.T, b). This uses LAPACK LU in
        # float64, as opposed to an inverse or pseudoinverse approximation.
        solved = np.linalg.solve(matrices[finite].transpose(0, 2, 1), difference[finite, :, None])[..., 0]
        quadratic = np.sum(solved * difference[finite], axis=1)
        if np.any(quadratic < 0):
            raise ArithmeticError("Original FADE distance has negative quadratic form")
        result[finite] = np.sqrt(quadratic)
    return result
