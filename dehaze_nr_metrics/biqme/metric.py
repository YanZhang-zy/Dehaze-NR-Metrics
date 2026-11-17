"""Native NumPy/SciPy BIQME image features and original SVR regression.

All 17 mathematical features are computed in Python. The original published
LIBSVM model/ranges remain read-only assets; no MATLAB process is launched.
The feature implementation follows the paper and public constituent
algorithms. It was checked on six small original-MATLAB oracle examples,
including images with odd sizes and dimensions not divisible by eight.

Primary sources:
    https://kegu.netlify.app/CODE/CODE_2018TNNLS.rar
    https://arxiv.org/abs/1904.08632
    https://github.com/cjlin1/libsvm/blob/master/svm-scale.c
    https://github.com/cjlin1/libsvm/blob/master/svm.cpp
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np


FEATURE_COUNT = 17
OFFICIAL_PACKAGE_URL = 'https://kegu.netlify.app/CODE/CODE_2018TNNLS.rar'
from ..common.models import resolve_models


class BIQMEUnavailableError(RuntimeError):
    """Reserved for callers using an unavailable BIQME asset or backend."""


def _features(values):
    array = np.asarray(values, dtype=np.float64)
    scalar = array.ndim == 1
    if scalar:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != FEATURE_COUNT or not len(array):
        raise ValueError('Expected 17 BIQME features or a nonempty N x 17 array')
    if not np.isfinite(array).all():
        raise ValueError('BIQME features must all be finite')
    return array, scalar


def _decimal_roundtrip(values, format_spec):
    # The original executable interface writes raw features with %f and
    # svm-scale writes scaled values with %g.  Those decimal conversions are
    # part of that historical interface, not an alternative image feature set.
    flat = np.fromiter((float(format(float(v), format_spec))
                        for v in values.flat), dtype=np.float64,
                       count=values.size)
    return flat.reshape(values.shape)


@dataclass(frozen=True)
class BIQMERange:
    """Published LIBSVM scaling ranges; values outside them are not clipped."""

    lower: float
    upper: float
    minima: np.ndarray
    maxima: np.ndarray

    @classmethod
    def load(cls, path):
        lines = Path(path).read_text(encoding='ascii').splitlines()
        if len(lines) != FEATURE_COUNT + 2 or lines[0].strip() != 'x':
            raise ValueError('Expected the 17-feature BIQME LIBSVM range file')
        bounds = lines[1].split()
        if len(bounds) != 2:
            raise ValueError('Invalid LIBSVM scaling bounds')
        lower, upper = map(float, bounds)
        minima = np.empty(FEATURE_COUNT, dtype=np.float64)
        maxima = np.empty(FEATURE_COUNT, dtype=np.float64)
        for expected, line in enumerate(lines[2:], 1):
            index, minimum, maximum = line.split()
            if int(index) != expected:
                raise ValueError('BIQME range indices must be consecutive 1..17')
            minima[expected - 1] = float(minimum)
            maxima[expected - 1] = float(maximum)
        if (not np.isfinite([lower, upper]).all() or upper <= lower
                or not np.isfinite(minima).all()
                or not np.isfinite(maxima).all()
                or not np.all(maxima > minima)):
            raise ValueError('Invalid or degenerate BIQME feature ranges')
        minima.setflags(write=False)
        maxima.setflags(write=False)
        return cls(lower, upper, minima, maxima)

    def transform(self, values, *, text_compat=True):
        """Apply original scaling, optionally reproducing its text roundtrips.

        ``text_compat=True`` follows raw %f (six decimals) then scaled %g
        (six significant digits).  False keeps the same linear scaling in
        float64 and is useful when the caller already has precise features.
        Neither mode clips out-of-training-range values.
        """
        array, scalar = _features(values)
        if text_compat:
            array = _decimal_roundtrip(array, '.6f')
        scaled = (self.lower + (self.upper - self.lower)
                  * (array - self.minima) / (self.maxima - self.minima))
        # libsvm explicitly preserves the exact endpoints.
        scaled = np.where(array == self.minima, self.lower, scaled)
        scaled = np.where(array == self.maxima, self.upper, scaled)
        if text_compat:
            scaled = _decimal_roundtrip(scaled, '.6g')
        return scaled[0] if scalar else scaled


@dataclass(frozen=True)
class BIQMEModel:
    """Native evaluator for the author's 17-feature RBF epsilon-SVR model.

    This class handles the regression stage. For image inputs use ``biqme``
    or compute their 17 features with ``extract_biqme_features`` first.
    """

    scale: BIQMERange
    support_vectors: np.ndarray
    coefficients: np.ndarray
    gamma: float
    rho: float
    assets_dir: Path

    @classmethod
    def load(cls, assets_dir=None):
        assets = resolve_models("biqme", assets_dir)
        scale = BIQMERange.load(assets / 'range')
        header = {}
        with (assets / 'model').open(encoding='ascii') as stream:
            for line in stream:
                if line.strip() == 'SV':
                    break
                key, *values = line.split()
                if not key or key in header:
                    raise ValueError('Malformed LIBSVM model header')
                header[key] = values
            else:
                raise ValueError('LIBSVM model has no support vectors')
            if (header.get('svm_type') != ['epsilon_svr']
                    or header.get('kernel_type') != ['rbf']
                    or header.get('nr_class') != ['2']):
                raise ValueError('Expected the BIQME RBF epsilon-SVR model')
            total = int(header['total_sv'][0])
            gamma = float(header['gamma'][0])
            rho = float(header['rho'][0])
            if total <= 0 or not np.isfinite([gamma, rho]).all() or gamma <= 0:
                raise ValueError('Invalid LIBSVM model parameters')
            vectors = np.zeros((total, FEATURE_COUNT), dtype=np.float64)
            coefficients = np.empty(total, dtype=np.float64)
            seen = 0
            for row, line in enumerate(stream):
                if row >= total:
                    raise ValueError('Too many support vectors in LIBSVM model')
                items = line.split()
                if not items:
                    raise ValueError('Empty LIBSVM support vector')
                coefficients[row] = float(items[0])
                previous = 0
                for item in items[1:]:
                    index_text, value_text = item.split(':')
                    index = int(index_text)
                    if not previous < index <= FEATURE_COUNT:
                        raise ValueError('Invalid BIQME support-vector index')
                    vectors[row, index - 1] = float(value_text)
                    previous = index
                seen += 1
        if (seen != total or not np.isfinite(vectors).all()
                or not np.isfinite(coefficients).all()):
            raise ValueError('Incomplete or non-finite BIQME support vectors')
        vectors.setflags(write=False)
        coefficients.setflags(write=False)
        return cls(scale, vectors, coefficients, gamma, rho, assets)

    def predict_scaled_features(self, values):
        """Evaluate already scaled feature rows using float64 RBF kernels."""
        array, scalar = _features(values)
        result = np.empty(len(array), dtype=np.float64)
        # One row at a time keeps scratch memory bounded for large batches.
        for row, features in enumerate(array):
            difference = self.support_vectors - features
            square_distance = np.einsum('ij,ij->i', difference, difference)
            result[row] = (np.dot(self.coefficients,
                                  np.exp(-self.gamma * square_distance))
                           - self.rho)
        if not np.isfinite(result).all():
            raise ValueError('BIQME prediction is non-finite')
        return float(result[0]) if scalar else result

    def predict_features(self, values, *, text_compat=True):
        """Scale exact BIQME features and predict; no MATLAB process needed."""
        return self.predict_scaled_features(
            self.scale.transform(values, text_compat=text_compat))

    def describe(self):
        return {
            'backend': 'native-python-numpy', 'feature_count': FEATURE_COUNT,
            'image_feature_extractor_available': True,
            'svm_type': 'epsilon_svr', 'kernel_type': 'rbf',
            'support_vectors': len(self.coefficients),
            'gamma': self.gamma, 'rho': self.rho,
            'assets_dir': str(self.assets_dir),
            'model_sha256': _sha256(self.assets_dir / 'model'),
            'range_sha256': _sha256(self.assets_dir / 'range'),
        }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _cached_model(assets_dir):
    return BIQMEModel.load(assets_dir)


def predict_biqme_features(features, *, assets_dir=None, text_compat=True):
    """Convenient cached access to the exact native BIQME regression stage."""
    assets = str(resolve_models("biqme", assets_dir))
    return _cached_model(assets).predict_features(features,
                                                text_compat=text_compat)


def inspect_biqme_assets(assets_dir=None):
    """Describe native feature/regression support and original model assets."""
    model = BIQMEModel.load(assets_dir)
    result = model.describe()
    pcode = model.assets_dir / 'BIQME.p'
    result.update({
        'image_to_score_available': True,
        'regression_from_original_features_available': True,
        'blocked_stage': None,
        'reason': None,
        'official_package_url': OFFICIAL_PACKAGE_URL,
        'pcode_sha256': _sha256(pcode) if pcode.is_file() else None,
        'matlab_started_by_this_module': False,
        'feature_implementation': 'paper equations and public PC/CDF97/NSS algorithms; small-sample oracle verified',
    })
    return result


def _histogram_entropy(gray):
    counts = np.bincount(gray.ravel(), minlength=256)
    probabilities = counts[counts > 0] / gray.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_biqme_features(image):
    """Compute the original model's 17 features from RGB uint8 image pixels.

    Paths and PIL images are also accepted. Float and 16-bit inputs require
    explicit conversion by the caller. This function does not call MATLAB
    and does not resize the RGB image globally: only PC and NSS use their
    original feature-specific averaging or half-resolution transforms.
    """
    from ..common.input import load_rgb_uint8
    from ..common.features import _contrast_energy, matlab_rgb2gray_uint8
    from .pc import pc_entropy
    from .dwt import wavelet_log_energies
    from .nss import nss_features

    rgb = load_rgb_uint8(image)
    if min(rgb.shape[:2]) < 16:
        raise ValueError('BIQME requires input height and width >= 16')
    gray = matlab_rgb2gray_uint8(rgb)
    gray_double = gray.astype(np.float64)
    brightness = []
    for multiplier in (1/3.5, 1/5.5, 1/7.5, 3.5, 5.5, 7.5):
        adjusted = np.clip(np.floor(gray_double * multiplier + .5), 0, 255).astype(np.uint8)
        brightness.append(_histogram_entropy(adjusted))
    contrast = [float(channel.mean()) for channel in _contrast_energy(rgb)]
    r, g, b = (rgb[..., channel].astype(np.float64) for channel in range(3))
    rg, yb = r-g, (r+g)/2-b
    colorfulness = (np.sqrt(np.var(rg, ddof=1) + np.var(yb, ddof=1))
                    + .3*np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
    maximum, minimum = rgb.max(axis=2).astype(np.float64), rgb.min(axis=2).astype(np.float64)
    saturation = np.divide(maximum-minimum, maximum,
                           out=np.zeros_like(maximum), where=maximum != 0)
    features = np.concatenate((
        [pc_entropy(gray)], wavelet_log_energies(gray), brightness,
        contrast, [np.mean(minimum/255), np.mean(saturation), colorfulness],
        nss_features(gray)))
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ValueError('BIQME produced invalid or non-finite image features')
    return features


def biqme(image, *, assets_dir=None, text_compat=True,
          return_features=False, return_diagnostics=False):
    """Compute native BIQME; larger values indicate better visual quality.

    The default reproduces original decimal %f -> svm-scale %g interfaces.
    ``return_features=True`` returns (score, features). Diagnostics additionally
    include exact model fingerprints and explicitly identify the Python
    backend. Neither option changes historical MATLAB benchmark results.
    """
    features = extract_biqme_features(image)
    assets = str(resolve_models("biqme", assets_dir))
    model = _cached_model(assets)
    score = model.predict_features(features, text_compat=text_compat)
    if return_diagnostics:
        return score, {'features': features, 'assets': model.describe(),
                       'text_compat': bool(text_compat),
                       'backend': 'native-python-numpy-scipy'}
    return (score, features) if return_features else score


__all__ = ['BIQMEUnavailableError', 'BIQMERange', 'BIQMEModel',
           'predict_biqme_features', 'inspect_biqme_assets',
           'extract_biqme_features', 'biqme']
