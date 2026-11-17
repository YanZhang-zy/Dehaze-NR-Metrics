"""Public BIQME metric."""
from ..common.interface import _finite, _pixels

__all__ = ["biqme"]


def biqme(image, *, model_dir=None):
    """Estimate image quality (higher = better); input must be at least 16 x 16."""
    pixels = _pixels(image)
    from .metric import biqme as calculate
    return _finite("biqme", calculate(pixels, assets_dir=model_dir))
