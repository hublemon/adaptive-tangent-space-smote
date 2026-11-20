"""Validation for the deliberately narrow dense, binary input contract."""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.utils.multiclass import check_classification_targets


def validate_data(X, y):
    if sparse.issparse(X):
        raise ValueError("X must be a dense real array; sparse input is not supported.")
    X = np.asarray(X)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty 2-dimensional array.")
    if X.dtype.kind not in "iuf":
        raise ValueError("X must contain real numeric features, not categorical or complex values.")
    X = np.array(X, dtype=np.float64, copy=True)
    if not np.isfinite(X).all():
        raise ValueError("X must be finite: NaN and infinity are not supported.")
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1-dimensional with the same number of rows as X.")
    if y.dtype.kind not in "biufUS":
        raise ValueError("y must contain numeric, boolean, or string class labels.")
    if y.dtype.kind in "f" and not np.isfinite(y).all():
        raise ValueError("y must contain finite labels, without NaN or infinity.")
    check_classification_targets(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Exactly two classes are required; multiclass input is not supported.")
    return X, y.copy(), classes, counts


def validate_parameters(estimator):
    for name in ("k_neighbors", "max_dim", "max_attempts"):
        value = getattr(estimator, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("variance_threshold", "safety_ratio"):
        value = getattr(estimator, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be a finite number in (0, 1].")
    if estimator.fallback not in ("duplicate", "raise"):
        raise ValueError("fallback must be 'duplicate' or 'raise'.")
    for name in ("adapt_dimension", "check_boundary", "allow_fallback"):
        if not isinstance(getattr(estimator, name), (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean.")
    seed = estimator.random_state
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("random_state must be None or a nonnegative integer.")


def generation_count(strategy, classes, counts):
    index = int(np.argmin(counts))
    minority = classes[index]
    n_min, n_maj = int(counts[index]), int(counts[1 - index])
    if isinstance(strategy, str) and strategy == "auto":
        target = n_maj
    elif isinstance(strategy, Mapping):
        if len(strategy) != 1 or minority not in strategy:
            raise ValueError("sampling_strategy dict must contain only the minority class.")
        target = strategy[minority]
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, Integral):
            raise ValueError("sampling_strategy target must be an integer final class count.")
    elif isinstance(strategy, Real) and not isinstance(strategy, (bool, np.bool_)):
        if not np.isfinite(strategy) or not 0 < strategy <= 1:
            raise ValueError("sampling_strategy ratio must be in (0, 1].")
        target = int(float(strategy) * n_maj)
    else:
        raise ValueError("sampling_strategy must be 'auto', a float ratio, or a minority dict.")
    if target < n_min:
        raise ValueError("sampling_strategy would remove minority rows; oversampling only.")
    return minority, int(target - n_min)
