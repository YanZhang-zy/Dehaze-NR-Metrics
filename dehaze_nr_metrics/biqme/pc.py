"""Phase-congruency entropy feature for the published BIQME model.

This is a native implementation of equations 1--2 in Gu et al.'s BIQME
paper, using Kovesi's public phasecong3 filter bank. It does not load, inspect,
or execute the author's P-code. Caller supplies the full-resolution 8-bit
MATLAB-compatible grayscale image; internal averaging and subsampling belong
to the phase-congruency feature alone.

References:
https://arxiv.org/abs/1904.08632
https://www.peterkovesi.com/matlabfns/PhaseCongruency/phasecong3.m

The filter-bank and phase-congruency equations follow Peter Kovesi's code.
Copyright (c) 1996-2017 Peter Kovesi
www.peterkovesi.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

The Software is provided "as is", without warranty of any kind.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _pc_input(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float64)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError('Expected a nonempty, finite 2D grayscale image')
    if image.min() < 0 or image.max() > 255:
        raise ValueError('Expected grayscale intensities in the range 0..255')
    factor = max(1, int(np.floor(min(image.shape) / 256.0 + 0.5)))
    if factor > 1:
        kernel = np.full((factor, factor), 1.0 / factor**2)
        # MATLAB imfilter correlation has its anchor at floor((size+1)/2).
        image = ndimage.correlate(image, kernel, mode='reflect',
                                  origin=-1 if factor % 2 == 0 else 0)
    return image[::factor, ::factor]


def phase_congruency_max_moment(gray: np.ndarray) -> np.ndarray:
    """Maximum phase-congruency covariance moment of a prepared gray image.

    ``gray`` is the internally averaged/subsampled image, not RGB. The default
    odd-dimension grid is the one used by BIQME's published reference package.
    """
    image = np.asarray(gray, dtype=np.float64)
    rows, cols = image.shape
    if min(rows, cols) < 2:
        raise ValueError('Phase congruency requires at least two rows and columns')
    def coordinate(n):
        denominator = n - 1 if n % 2 else n
        return (np.arange(n) - n // 2) / denominator
    x, y = np.meshgrid(coordinate(cols), coordinate(rows))
    radius = np.fft.ifftshift(np.sqrt(x*x + y*y))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    lowpass = 1.0 / (1.0 + (radius / .45)**30)
    radius[0, 0] = 1.0
    sine, cosine = np.sin(theta), np.cos(theta)
    radial = []
    for scale in range(4):
        wavelength = 3.0 * 2.1**scale
        kernel = np.exp(-np.log(radius * wavelength)**2 / (2 * np.log(.55)**2))
        kernel *= lowpass
        kernel[0, 0] = 0.0
        radial.append(kernel)
    image_fft = np.fft.fft2(image)
    cov_xx = np.zeros_like(image)
    cov_yy = np.zeros_like(image)
    cov_xy = np.zeros_like(image)
    epsilon = 1e-4
    for orientation in range(6):
        angle = orientation * np.pi / 6
        ds = sine * np.cos(angle) - cosine * np.sin(angle)
        dc = cosine * np.cos(angle) + sine * np.sin(angle)
        distance = np.minimum(np.abs(np.arctan2(ds, dc)) * 3, np.pi)
        angular = (np.cos(distance) + 1) / 2
        responses = [np.fft.ifft2(image_fft * kernel * angular) for kernel in radial]
        amplitudes = [np.abs(response) for response in responses]
        sum_amplitude = sum(amplitudes)
        sum_even = sum(response.real for response in responses)
        sum_odd = sum(response.imag for response in responses)
        mean_norm = np.sqrt(sum_even**2 + sum_odd**2) + epsilon
        mean_even, mean_odd = sum_even / mean_norm, sum_odd / mean_norm
        energy = sum(response.real * mean_even + response.imag * mean_odd
                     - np.abs(response.real * mean_odd - response.imag * mean_even)
                     for response in responses)
        tau = np.median(amplitudes[0]) / np.sqrt(np.log(4))
        total_tau = tau * (1 - (1/2.1)**4) / (1 - 1/2.1)
        threshold = total_tau * (np.sqrt(np.pi/2) + 2 * np.sqrt((4-np.pi)/2))
        energy = np.maximum(energy - threshold, 0)
        width = (sum_amplitude / (np.maximum.reduce(amplitudes) + epsilon) - 1) / 3
        weight = 1.0 / (1.0 + np.exp((.5 - width) * 10))
        pc = np.divide(weight * energy, sum_amplitude,
                       out=np.zeros_like(energy), where=sum_amplitude != 0)
        cov_x, cov_y = pc * np.cos(angle), pc * np.sin(angle)
        cov_xx += cov_x**2
        cov_yy += cov_y**2
        cov_xy += cov_x * cov_y
    cov_xx /= 3
    cov_yy /= 3
    cov_xy *= 4/6
    denominator = np.sqrt(cov_xy**2 + (cov_xx - cov_yy)**2) + epsilon
    return (cov_xx + cov_yy + denominator) / 2


def pc_entropy(gray: np.ndarray, *, return_debug=False):
    """BIQME entropy at the round(40% * N) largest PC maximum moments."""
    prepared = _pc_input(gray)
    moment = phase_congruency_max_moment(prepared)
    # MATLAB round uses half away from zero; ceil would differ on aerial.png.
    count = int(np.floor(.4 * prepared.size + .5))
    positions = np.argsort(-moment.ravel(order='F'), kind='stable')[:count]
    gray_uint8 = np.clip(np.floor(prepared + .5), 0, 255).astype(np.uint8)
    samples = gray_uint8.ravel(order='F')[positions]
    histogram = np.bincount(samples, minlength=256)
    probabilities = histogram[histogram > 0] / samples.size
    result = float(-np.sum(probabilities * np.log2(probabilities)))
    if return_debug:
        return result, {'prepared_gray': prepared, 'pc_max_moment': moment,
                        'selected_gray': samples, 'selected_positions': positions}
    return result
