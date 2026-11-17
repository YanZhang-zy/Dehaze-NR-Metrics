"""FADE, BIQME and RGB entropy with a shared, explicit image interface."""
from __future__ import annotations

from .biqme import biqme
from .entropy import entropy
from .fade import fade
from .common.interface import _pixels

__version__ = "0.1.0"
METRICS = ("fade", "biqme", "entropy")
__all__ = ["fade", "biqme", "entropy", "score_image", "score_folder"]


def _names(metrics):
    names = (metrics,) if isinstance(metrics, str) else tuple(metrics)
    if not names or len(set(names)) != len(names) or any(n not in METRICS for n in names):
        raise ValueError("Choose unique metrics from: fade, biqme, entropy")
    return names


def score_image(image, metrics=METRICS):
    """Return a {metric: float} dictionary; raise on an invalid image or score.

    ``metrics`` can be one name or a sequence of unique names. No resize,
    range normalization, EXIF rotation, network request or GPU is used.
    """
    names = _names(metrics)
    pixels = _pixels(image)
    functions = {"fade": fade, "biqme": biqme, "entropy": entropy}
    return {name: functions[name](pixels) for name in names}


def score_folder(folder, output, metrics=METRICS):
    """Score immediate image files into a new directory and return its summary.

    Writes per_image.csv, summary.csv and summary.json. Failed scores remain
    in per_image.csv and are excluded from each metric's mean. Inspect the
    returned ``complete`` field. An existing output directory is an error.
    """
    from .batch import evaluate
    return evaluate(folder, output, metrics=metrics, folder_only=True)
