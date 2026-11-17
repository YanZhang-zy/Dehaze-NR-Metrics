"""Portable Python implementation of the calibrated FADE Gaussian filter.

No custom C source, compiler, ctypes, shared library or OS branch is used.
The same required Numba version compiles this Python kernel with explicit LLVM
FMA on both Linux and Windows. There is one production implementation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from ..common.runtime import RELEASE_ID, require_fixed_runtime

ALGORITHM_ID = 'fade_windows_r2023a_gaussian_fixed_numba_v1'
_COL = tuple(float.fromhex(x) for x in (
    '-0x1.9b92991f24884p-7', '-0x1.42e11ca517a60p-4',
    '-0x1.e5fb7c557fad1p-3', '-0x1.5edacbc602378p-2',
    '-0x1.e5fb7c557fad1p-3', '-0x1.42e11ca517a60p-4',
    '-0x1.9b92991f24881p-7'))
_ROW = tuple(float.fromhex(x) for x in (
    '-0x1.9b92991f24880p-7', '-0x1.42e11ca517a5ep-4',
    '-0x1.e5fb7c557fad1p-3', '-0x1.5edacbc602378p-2',
    '-0x1.e5fb7c557fad1p-3', '-0x1.42e11ca517a5fp-4',
    '-0x1.9b92991f24881p-7'))
@lru_cache(maxsize=1)
def _numba_kernel():
    require_fixed_runtime()
    from numba import njit, types
    from numba.extending import intrinsic
    from llvmlite import ir

    @intrinsic
    def fused_multiply_add(typing_context, a, b, c):
        if any(t != types.float64 for t in (a, b, c)):
            return None
        signature = types.float64(a, b, c)
        def codegen(context, builder, sig, arguments):
            function = builder.module.globals.get('llvm.fma.f64')
            if function is None:
                double = ir.DoubleType()
                function = ir.Function(builder.module,
                                       ir.FunctionType(double, [double, double, double]),
                                       name='llvm.fma.f64')
            return builder.call(function, arguments)
        return signature, codegen

    @njit(cache=False, fastmath=False, nogil=True)
    def kernel(image):
        height, width = image.shape
        temporary = np.empty_like(image)
        output = np.empty_like(image)
        for row in range(height):
            for column in range(width):
                value = 0.0
                for k in range(7):
                    source = min(max(row + 3 - k, 0), height - 1)
                    if height == 16:
                        product = image[source, column] * _COL[k]
                        value = value + product
                    else:
                        value = fused_multiply_add(image[source, column], _COL[k], value)
                temporary[row, column] = value
        for row in range(height):
            for column in range(width):
                value = 0.0
                for k in range(7):
                    source = min(max(column + 3 - k, 0), width - 1)
                    value = fused_multiply_add(temporary[row, source], _ROW[k], value)
                output[row, column] = value
        return output

    kernel(np.zeros((16, 16), dtype=np.float64))
    return kernel


def gaussian_filter(image, *, cache_dir=None):
    """Fixed 7x7 filter: replicate edges, ordered operations and explicit FMA.

    cache_dir is accepted for API compatibility and unused. There is no
    platform selection, environment selector or alternative scoring backend.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError('Expected a nonempty, finite 2D array')
    return _numba_kernel()(np.ascontiguousarray(array))


def filter_provenance(*, cache_dir=None):
    versions = dict(require_fixed_runtime())
    _numba_kernel()
    return {'release_id': RELEASE_ID, 'algorithm_id': ALGORITHM_ID,
            'fixed_backend': 'numba', 'automatic_fallback': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'custom_c_compiler_required': False, 'custom_shared_library_loaded': False,
            'arithmetic': 'explicit FMA; height==16 vertical pass uses separate multiply/add',
            'fastmath': False, 'parallel': False, 'versions': versions}
