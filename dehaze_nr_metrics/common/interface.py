"""Shared validation for the public metric functions."""
import math
import os


def _pixels(image):
    from .runtime import require_fixed_runtime
    require_fixed_runtime()
    from .input import load_rgb_uint8
    return load_rgb_uint8(os.fspath(image) if isinstance(image, os.PathLike) else image)


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} returned a non-finite score; the image may be degenerate")
    return value
