"""Public FADE metric."""
from ..common.interface import _finite, _pixels

__all__ = ["fade"]


def fade(image, *, model_dir=None):
    """Estimate fog density (lower = less fog); input must be at least 16 x 16.

    Accept a path, PIL image, or uint8 NumPy array. Uses the calibrated PS8
    implementation, including the original internal crop to multiples of 8.
    """
    pixels = _pixels(image)
    from .metric import fade_windows_r2023a
    return _finite("fade", fade_windows_r2023a(pixels, model_dir=model_dir))
