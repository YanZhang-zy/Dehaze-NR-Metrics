"""Nonrecursive batch evaluation with explicit failures and reproducible output."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics

from . import METRICS, __version__, _names, _pixels, biqme, entropy, fade
from .common.models import manifest, resolve_models
from .common.runtime import require_fixed_runtime

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def evaluate(source, output, metrics=METRICS, *, folder_only=False):
    names = _names(metrics)
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    elif not folder_only and source.is_file() and source.suffix.lower() in EXTENSIONS:
        paths = [source]
    else:
        raise ValueError("Input must be an existing image folder" if folder_only
                         else "Input must be an image or an existing folder")
    if not paths:
        raise ValueError("No supported images in the input folder")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    versions = dict(require_fixed_runtime())
    for name in names:
        if name in manifest():
            resolve_models(name)
    output.mkdir(parents=True, exist_ok=False)
    functions = {"fade": fade, "biqme": biqme, "entropy": entropy}
    values = {name: [] for name in names}
    with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "metric", "score", "status", "error"])
        writer.writeheader()
        for path in paths:
            try:
                pixels, decode_error = _pixels(path), None
            except Exception as error:
                pixels, decode_error = None, error
            for name in names:
                row = {"image": path.name, "metric": name, "score": "", "status": "ok", "error": ""}
                try:
                    if decode_error is not None:
                        raise decode_error
                    row["score"] = functions[name](pixels)
                    values[name].append(row["score"])
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                writer.writerow(row)
            stream.flush()
    summaries = [{"metric": name, "count": len(values[name]), "total": len(paths),
                  "errors": len(paths) - len(values[name]),
                  "mean": statistics.fmean(values[name]) if values[name] else None,
                  "std_population": statistics.pstdev(values[name]) if values[name] else None}
                 for name in names]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "complete": all(not item["errors"] for item in summaries),
        "images": len(paths), "metrics": summaries, "version": __version__,
        "python": platform.python_version(), "platform": platform.system(),
        "versions": versions, "backend": "python-cpu", "folder_scope": "immediate_directory_only",
        "external_resize": False, "input": source.name,
        "fade_algorithm": "original_fade_ps8_windows_r2023a_compatible_v1" if "fade" in names else None,
        "models_sha256": {name: manifest()[name] for name in names if name in manifest()},
        "per_image_csv_sha256": hashlib.sha256((output / "per_image.csv").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
