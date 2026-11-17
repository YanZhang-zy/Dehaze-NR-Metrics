"""Native NSS features used by BIQME (GGD shape and second moment).

The published BIQME naturalness model locally normalizes a grayscale image
and fits the shape of a generalized Gaussian distribution.  This module is
a clean implementation of those mathematical operations, verified against
the author's callable BIQME package; it does not load or execute MATLAB.

The caller supplies the original 0..255 grayscale image.  In particular,
resizing is performed in double precision without uint8 quantization.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma


_HALF_WEIGHTS = np.array([-3., -9., 29., 111., 111., 29., -9., -3.]) / 256.


def matlab_bicubic_half(gray: np.ndarray) -> np.ndarray:
    """Antialiased bicubic 0.5x scaling with MATLAB's symmetric boundaries.

    A half scale uses eight cubic-kernel taps and maps output coordinate j
    to input coordinate 2*j+0.5 (zero based).  The output shape is ceil(H/2)
    by ceil(W/2); both passes retain double precision.  uint8 input is not
    rounded or clipped between passes or on output.
    """
    result = np.asarray(gray, dtype=np.float64)
    if result.ndim != 2 or not result.size or not np.isfinite(result).all():
        raise ValueError('Expected a nonempty, finite 2D grayscale image')
    for axis in (0, 1):
        length = result.shape[axis]
        positions = (np.arange((length + 1) // 2)[:, None] * 2
                     + np.arange(-3, 5)[None, :])
        # Repeating the two edge pixels is MATLAB's symmetric convention.
        reflected = np.mod(positions, 2 * length)
        indices = np.where(reflected < length, reflected,
                           2 * length - 1 - reflected)
        moved = np.moveaxis(result, axis, 0)
        reduced = np.zeros((len(positions), *moved.shape[1:]), dtype=np.float64)
        for tap, weight in enumerate(_HALF_WEIGHTS):
            reduced += weight * moved[indices[:, tap]]
        result = np.moveaxis(reduced, 0, axis)
    return result


@lru_cache(maxsize=1)
def _ggd_lookup() -> tuple[np.ndarray, np.ndarray]:
    shapes = 0.2 + np.arange(9801, dtype=np.float64) * 0.001
    ratios = gamma(1. / shapes) * gamma(3. / shapes) / gamma(2. / shapes) ** 2
    return shapes, ratios


@lru_cache(maxsize=1)
def _local_window() -> np.ndarray:
    coordinate = np.arange(-3., 4.)
    kernel = np.exp(-(coordinate[:, None] ** 2 + coordinate[None, :] ** 2)
                    / (2 * (7. / 6.) ** 2))
    kernel /= kernel.sum()
    return kernel


def nss_features(gray: np.ndarray) -> np.ndarray:
    """Return BIQME features 16 and 17: GGD shape and MSCN second moment.

    ``gray`` must be the original grayscale image in the 0..255 intensity
    convention.  For RGB uint8, first use ``matlab_rgb2gray_uint8`` from
    ``iqa_suite.native.fade``.  The 7x7 Gaussian window has sigma=7/6 and
    zero padding, matching ``filter2(..., 'same')``.  The scale denominator
    is local standard deviation plus 1.  The second feature is mean(MSCN^2),
    rather than sample variance about the global MSCN mean.
    """
    resized = matlab_bicubic_half(gray)
    window = _local_window()
    local_mean = convolve(resized, window, mode='constant', cval=0.)
    local_second = convolve(resized ** 2, window, mode='constant', cval=0.)
    local_std = np.sqrt(np.abs(local_second - local_mean ** 2))
    normalized = (resized - local_mean) / (local_std + 1.)
    second_moment = float(np.mean(normalized ** 2))
    mean_absolute = float(np.mean(np.abs(normalized)))
    shapes, ratios = _ggd_lookup()
    if mean_absolute == 0:
        # The zero/zero estimate is degenerate. The first lookup position
        # follows MATLAB's minimum-index behavior for an all-NaN distance.
        shape = shapes[0]
    else:
        observed_ratio = second_moment / mean_absolute ** 2
        shape = shapes[np.argmin(np.abs(ratios - observed_ratio))]
    return np.array([shape, second_moment], dtype=np.float64)


__all__ = ['matlab_bicubic_half', 'nss_features']
