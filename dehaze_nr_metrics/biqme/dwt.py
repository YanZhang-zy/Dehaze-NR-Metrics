"""CDF 9/7 wavelet log energies used by the BIQME image features.

The lifting and odd-length extrapolation equations follow Pascal Getreuer's
public mathematical description, https://getreuer.info/posts/waveletcdf97/.
This module implements the equations independently in NumPy.  BIQME's band
selection uses floor(original_dimension / 2**level), which differs from the
usual ceil-sized approximation bands for images not divisible by eight.
"""
from __future__ import annotations

import numpy as np

_A = -1.5861343420693648
_B = -0.0529801185718856
_C = 0.8829110755411875
_D = 0.4435068520511142
_K = 1.1496043988602418


def _lift_axis(image: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.moveaxis(image, axis, -1)
    length = signal.shape[-1]
    if length < 2:
        raise ValueError('CDF 9/7 lifting requires at least two samples')
    odd_length = bool(length % 2)
    if odd_length:
        # Extrapolate so that the appended sample's final highpass value is
        # zero. Repeating the edge would change BIQME's original features.
        if length < 3:
            raise ValueError('Odd CDF 9/7 lifting needs at least three samples')
        extrapolated = (-2 / (1 + 2 * _B * _C)) * (
            _A * _B * _C * signal[..., -3:-2]
            + _B * _C * signal[..., -2:-1]
            + (_A + 3 * _A * _B * _C + _C) * signal[..., -1:])
        signal = np.concatenate((signal, extrapolated), axis=-1)
    low = signal[..., ::2].copy()
    high = signal[..., 1::2].copy()
    for coefficient, predict in ((_A, True), (_B, False),
                                 (_C, True), (_D, False)):
        if predict:
            right = np.concatenate((low[..., 1:], low[..., -1:]), axis=-1)
            high += coefficient * (low + right)
        else:
            left = np.concatenate((high[..., :1], high[..., :-1]), axis=-1)
            low += coefficient * (left + high)
    low *= _K
    high /= _K
    if odd_length:
        high = high[..., :-1]
    return np.moveaxis(low, -1, axis), np.moveaxis(high, -1, axis)


def wavelet_coefficients(gray: np.ndarray, levels: int = 3) -> np.ndarray:
    """Tensor CDF 9/7 transform with nonredundant half-sample boundaries."""
    result = np.asarray(gray, dtype=np.float64).copy()
    if (result.ndim != 2 or min(result.shape) < 2**levels
            or not np.isfinite(result).all()):
        raise ValueError('Expected finite grayscale with height/width >= 2**levels')
    height, width = result.shape
    for _ in range(levels):
        horizontal_low, horizontal_high = _lift_axis(result[:height, :width], 1)
        ll, lh = _lift_axis(horizontal_low, 0)
        hl, hh = _lift_axis(horizontal_high, 0)
        result[:height, :width] = np.block([[ll, hl], [lh, hh]])
        height, width = (height + 1) // 2, (width + 1) // 2
    return result


def wavelet_log_energies(gray: np.ndarray) -> np.ndarray:
    """Return BIQME's second and third wavelet-level sharpness features."""
    gray = np.asarray(gray, dtype=np.float64)
    coefficients = wavelet_coefficients(gray, levels=3)
    height, width = gray.shape
    energies = []
    for level in (2, 3):
        outer_height, outer_width = height // 2**(level-1), width // 2**(level-1)
        inner_height, inner_width = height // 2**level, width // 2**level
        bands = (coefficients[inner_height:outer_height, :inner_width],
                 coefficients[:inner_height, inner_width:outer_width],
                 coefficients[inner_height:outer_height, inner_width:outer_width])
        logs = [np.log10(1 + np.mean(band**2)) for band in bands]
        energies.append(((logs[0] + logs[1])/2 + 4*logs[2])/5)
    return np.asarray(energies, dtype=np.float64)


__all__ = ['wavelet_coefficients', 'wavelet_log_energies']
