"""Public RGB entropy metric."""
from ..common.interface import _finite, _pixels

__all__ = ["entropy"]


def entropy(image):
    """Return Shannon entropy in bits over the pooled 256-bin RGB histogram."""
    pixels = _pixels(image)
    from .metric import entropy as calculate
    return _finite("entropy", calculate(pixels))
