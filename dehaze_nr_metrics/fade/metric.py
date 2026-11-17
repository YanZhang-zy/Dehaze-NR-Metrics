"""Calibrated FADE with a portable Python implementation of its filter.

The feature equations are unchanged. Fixed, image-independent SVD coefficients
and explicit fused multiply-add ordering are preserved by the same required
Numba kernel on Linux and Windows. No custom C kernel, MATLAB, golden arrays or score lookup is used.
The Fattal comparison with MATLAB R2023a is recorded in README.md.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from ..common.features import _contrast_energy, matlab_rgb2gray_uint8
from .features import (_distances, _nanvar_rows, _patch_entropy, _patches,
                       _rgb_uint8, FEATURE_NAMES, load_models, model_fingerprints)
from .filter import gaussian_filter, filter_provenance


ALGORITHM_ID = 'original_fade_ps8_windows_r2023a_compatible_v1'


def extract_features(image, *, cache_dir=None):
    """Original FADE feature equations with an isolated Gaussian filter call."""
    rgb = _rgb_uint8(image)
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    gray_uint8 = matlab_rgb2gray_uint8(rgb)
    gray = gray_uint8.astype(np.float64)
    mu = gaussian_filter(gray, cache_dir=cache_dir)
    second_moment = gaussian_filter(gray * gray, cache_dir=cache_dir)
    sigma = np.sqrt(np.abs(second_moment - mu * mu))
    mscn = (gray - mu) / (sigma + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = sigma / mu
    paired = _patches(mscn * np.roll(mscn, 1, axis=0))
    paired_left = np.where(paired > 0, np.nan, paired)
    paired_right = np.where(paired < 0, np.nan, paired)
    ce_gray, ce_by, ce_rg = _contrast_energy(rgb)
    rg, by = _patches(r - g), _patches(0.5 * (r + g) - b)
    colorfulness = np.sqrt(np.var(rg, axis=1, ddof=1) + np.var(by, axis=1, ddof=1)) + 0.3 * np.sqrt(rg.mean(axis=1) ** 2 + by.mean(axis=1) ** 2)
    maximum, minimum = rgb.max(axis=2).astype(np.float64), rgb.min(axis=2).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        saturation = (maximum - minimum) / maximum
    saturation[maximum == 0] = 0
    features = np.column_stack((
        _nanvar_rows(_patches(mscn)), _nanvar_rows(paired_right), _nanvar_rows(paired_left),
        _patches(sigma).mean(axis=1), _patches(cv).mean(axis=1),
        _patches(ce_gray).mean(axis=1), _patches(ce_by).mean(axis=1), _patches(ce_rg).mean(axis=1),
        _patch_entropy(gray_uint8), _patches(minimum / 255.0).mean(axis=1),
        _patches(saturation).mean(axis=1), colorfulness,
    ))
    features = np.log(1 + features)
    return features, {'gray_image': gray, 'mu': mu, 'second_moment': second_moment,
                      'sigma': sigma, 'mscn': mscn,
                      'ce_gray': ce_gray, 'ce_by': ce_by, 'ce_rg': ce_rg,
                      'patch_shape': (rgb.shape[0] // 8, rgb.shape[1] // 8)}


def fade_windows_r2023a(path_or_rgb, *, model_dir: str | Path | None = None,
                       cache_dir: str | Path | None = None,
                       return_map=False, return_diagnostics=False):
    """Compute calibrated FADE on original uint8 RGB without a custom C library.

    Return a score, ``(score, map)`` or ``(score, map, trace)``. The filter is fixed to the same Python/Numba kernel on both platforms.
    cache_dir is accepted for API compatibility and is unused.
    """
    features, trace = extract_features(path_or_rgb, cache_dir=cache_dir)
    patch_variance = _nanvar_rows(features)
    fogfree, foggy = load_models(model_dir)
    distance_free = _distances(features, fogfree, patch_variance)
    distance_fog = _distances(features, foggy, patch_variance)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        df, dff = np.nanmean(distance_free), np.nanmean(distance_fog)
    score = float(df / (dff + 1))
    density_map = (distance_free / (distance_fog + 1)).reshape(trace['patch_shape'], order='F')
    if return_diagnostics:
        trace.update(feature_matrix=features, feature_names=FEATURE_NAMES,
                     patch_variance=patch_variance, distance_fogfree=distance_free,
                     distance_foggy=distance_fog, Df=float(df), Dff=float(dff),
                     models=model_fingerprints(model_dir), dtype='float64',
                     algorithm_id=ALGORITHM_ID, filter=filter_provenance(cache_dir=cache_dir))
        return score, density_map, trace
    return (score, density_map) if return_map else score


__all__ = ['fade_windows_r2023a', 'extract_features', 'ALGORITHM_ID']
