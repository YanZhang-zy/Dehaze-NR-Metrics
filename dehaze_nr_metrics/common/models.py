"""Resolve bundled model parameters and verify their SHA-256 identities."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@lru_cache(maxsize=1)
def manifest():
    return json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=128)
def _verify(path, expected, size, mtime_ns):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError(
            f"Model SHA-256 mismatch: {path}. Restore the bundled model files "
            "or check your custom model directory."
        )


def resolve_models(metric, directory=None):
    """Return verified bundled files, or files in an explicit override.

    No downloads, cache setup or model import are performed.
    """
    if metric not in manifest():
        raise ValueError("Choose a model metric: fade or biqme")
    if directory is not None:
        root = Path(directory).expanduser()
    elif os.environ.get("DEHAZE_NR_METRICS_MODEL_DIR"):
        root = Path(os.environ["DEHAZE_NR_METRICS_MODEL_DIR"]).expanduser() / metric
    else:
        root = Path(__file__).resolve().parents[1] / metric / "data"
    root = root.resolve()
    for name, expected in manifest()[metric].items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {metric.upper()} model: {path}. "
                "Restore the bundled model files or check your custom model directory."
            )
        stat = path.stat()
        _verify(str(path), expected, stat.st_size, stat.st_mtime_ns)
    return root
