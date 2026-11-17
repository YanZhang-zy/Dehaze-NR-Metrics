"""Run dehaze-nr-metrics; the original model parameters are bundled."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    argv = list(sys.argv[1:] if argv is None else argv)
    from . import METRICS, __version__, _names, score_image
    parser = argparse.ArgumentParser(
        description="Compute FADE, BIQME and RGB entropy on the CPU.")
    parser.add_argument("input", type=Path, help="Image file or folder (immediate files only)")
    parser.add_argument("-o", "--output", type=Path, help="New results directory; required for folders")
    parser.add_argument("-m", "--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    try:
        _names(args.metrics)
        if args.input.is_dir() and args.output is None:
            parser.error("Folder input requires -o <new-results-directory>")
        if args.output is not None:
            from .batch import evaluate
            result = evaluate(args.input, args.output, args.metrics)
        else:
            result = score_image(args.input, args.metrics)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("complete", True) else 1
    except (ValueError, TypeError, OSError, RuntimeError, ArithmeticError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
