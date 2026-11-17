"""The single numerical dependency lock for Linux and Windows."""
from functools import lru_cache
import importlib.metadata
import struct
import sys

RELEASE_ID = 'fade-biqme-entropy-unified-20261028-v1'
REQUIRED_VERSIONS = {
    'numpy': '1.26.4',
    'scipy': '1.15.3',
    'Pillow': '12.0.0',
    'numba': '0.62.1',
    'llvmlite': '0.45.1',
}


@lru_cache(maxsize=1)
def require_fixed_runtime():
    """Fail explicitly rather than changing implementation or dependencies."""
    if struct.calcsize('P') != 8 or sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        raise RuntimeError('Use 64-bit Python 3.10, 3.11 or 3.12 (3.10 recommended).')
    actual = {}
    errors = []
    for name, required in REQUIRED_VERSIONS.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != required:
            errors.append(f'{name}: required {required}, installed {actual[name]}')
    if errors:
        raise RuntimeError('Unified runtime version mismatch. Run python -m pip install -r requirements.txt '
                           'from the project directory. ' + '; '.join(errors))
    return actual
