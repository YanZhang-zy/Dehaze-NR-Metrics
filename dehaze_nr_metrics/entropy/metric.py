"""Python equivalent of MATLAB entropy(RGB uint8), measured in bits."""
import numpy as np

from ..common.input import load_rgb_uint8


def entropy(image):
    """Shannon entropy over the pooled 256-bin RGB histogram.

    The original benchmark calls entropy(I) on all RGB values. It does not
    first form a grayscale image and does not average per-channel entropies.
    """
    pixels = load_rgb_uint8(image)
    counts = np.bincount(pixels.reshape(-1), minlength=256)
    probabilities = counts[counts != 0].astype(np.float64) / pixels.size
    return float(-np.sum(probabilities * np.log2(probabilities)))
