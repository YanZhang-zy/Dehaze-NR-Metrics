"""Explicit RGB uint8 inputs shared by native reference metric ports."""
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_uint8(image):
    """Read 8-bit RGB pixels, expanding gray/palette and dropping alpha.

    Float and 16-bit inputs require an explicit caller conversion. Refusing an
    implicit range conversion avoids accidentally passing 0..1 data to FADE's
    0..255 feature equations.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return load_rgb_uint8(source)
    if isinstance(image, Image.Image):
        if image.mode not in {'1', 'L', 'LA', 'P', 'PA', 'RGB', 'RGBA'}:
            raise ValueError('Native metrics require 8-bit RGB/gray/palette input; convert explicitly first')
        array = np.asarray(image.convert('RGB'), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise TypeError('Expected uint8 pixels in 0..255; floating-point and uint16 conversion must be explicit')
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] in (3, 4):
            array = array[..., :3]
        else:
            raise ValueError('Expected HxW, HxWx1, HxWx3 or HxWx4 pixels')
    if not array.size:
        raise ValueError('Image is empty')
    return np.ascontiguousarray(array)
