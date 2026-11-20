"""Bounded generation with an explicit segment/duplicate fallback."""

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator

from .geometry import local_geometry, segment_candidate, tangent_candidate
from .validation import generation_count, validate_data, validate_parameters


class SamplingError(ValueError):
    """Requested count could not be safely synthesized under the chosen policy."""


class AdaptiveTangentSpaceSMOTE(BaseEstimator):
    """Oversample finite dense binary data using locally selected tangent spaces.

    Parameters
    ----------
    sampling_strategy : 'auto', float, or dict, default='auto'
        Balance classes, request a final minority/majority ratio, or map the
        minority label to its final count. Existing rows are never removed.
    k_neighbors : int, default=8
        Maximum minority neighbors, clipped to the available count minus one.
    max_dim : int, default=3
        Maximum local dimension. One forces explicit segment interpolation.
    variance_threshold : float, default=0.9
        Fraction of local weighted variance used to select dimension.
    safety_ratio : float, default=0.8
        Accept only d_min + eps < safety_ratio * (d_maj + eps).
    max_attempts : int, default=20
        Maximum attempts per generation stage and requested row.
    fallback : {'duplicate', 'raise'}, default='duplicate'
        Policy after both tangent and segment generation fail. Duplicates
        preserve observed rows but are not claimed to satisfy the safety test.
    random_state : nonnegative int or None, default=None
        Seed of a private NumPy Generator; global RNG state is untouched.
    adapt_dimension, check_boundary, allow_fallback : bool, default=True
        Research ablations. Disabling adaptation fixes dimension to the
        supported rank capped by max_dim. Disabling boundary checks also
        disables majority-aware radius contraction. Disabling fallback raises
        instead of using segments or duplicates when tangent sampling fails.

    Attributes
    ----------
    diagnostics_ : dict
        Counts, attempts, rejections, dimensions and fallback reasons.
    sample_metadata_ : list of dict
        For each appended row: input anchor index, path, dimension and radius.
    local_dimensions_ : ndarray
        Selected dimension for each original minority row.
    """

    def __init__(
        self,
        sampling_strategy="auto",
        *,
        k_neighbors=8,
        max_dim=3,
        variance_threshold=0.9,
        safety_ratio=0.8,
        max_attempts=20,
        fallback="duplicate",
        random_state=None,
        adapt_dimension=True,
        check_boundary=True,
        allow_fallback=True,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.max_dim = max_dim
        self.variance_threshold = variance_threshold
        self.safety_ratio = safety_ratio
        self.max_attempts = max_attempts
        self.fallback = fallback
        self.random_state = random_state
        self.adapt_dimension = adapt_dimension
        self.check_boundary = check_boundary
        self.allow_fallback = allow_fallback

    def fit(self, X, y):
        """Validate input and initialize diagnostics without generating rows."""
        self._prepare(X, y)
        return self

    def _prepare(self, X, y):
        # Clear prior fitted state even if a repeated fit fails validation.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        validate_parameters(self)
        X, y, self.classes_, counts = validate_data(X, y)
        self.minority_class_, requested = generation_count(
            self.sampling_strategy, self.classes_, counts
        )
        self.n_features_in_ = X.shape[1]
        self.n_samples_in_ = len(X)
        self.sampling_strategy_ = {self.minority_class_: requested}
        self.sample_metadata_ = []
        self.diagnostics_ = dict(
            requested=requested,
            generated=0,
            tangent=0,
            segment=0,
            duplicate=0,
            attempts=0,
            boundary_rejections=0,
            radius_rejections=0,
            dimension_counts={},
            geometry_reasons={},
        )
        return X, y, requested

    def fit_resample(self, X, y):
        """Return original rows followed by exactly the requested new rows.

        All geometry is fitted exclusively on the supplied data. Put this
        sampler inside an imbalanced-learn Pipeline to avoid leakage.
        """
        X, y, requested = self._prepare(X, y)
        if not requested:
            self.local_dimensions_ = np.array([], dtype=int)
            return X, y
        rng = np.random.default_rng(self.random_state)
        minority_indices = np.flatnonzero(y == self.minority_class_)
        points = X[minority_indices]
        other = X[y != self.minority_class_]
        minority_tree, majority_tree = cKDTree(points), cKDTree(other)
        k = min(self.k_neighbors, len(points) - 1)
        # Explicit index exclusion is essential with duplicate coordinates.
        _, indices = minority_tree.query(points, k=k + 1)
        indices = np.asarray(indices).reshape(len(points), k + 1)
        geometries = []
        for i, row in enumerate(indices):
            neighbors = row[row != i][:k]
            geometries.append(
                local_geometry(
                    points,
                    i,
                    neighbors,
                    self.max_dim,
                    self.variance_threshold,
                    self.adapt_dimension,
                )
            )
        self.local_dimensions_ = np.array([g.dimension for g in geometries])
        self.diagnostics_["dimension_counts"] = dict(Counter(self.local_dimensions_.tolist()))
        self.diagnostics_["geometry_reasons"] = dict(Counter(g.reason for g in geometries))
        output = []
        for _ in range(requested):
            i = int(rng.integers(len(points)))
            anchor, geometry = points[i], geometries[i]
            majority_distance = float(majority_tree.query(anchor)[0])
            radius = geometry.radius
            if self.check_boundary:
                radius = min(radius, self.safety_ratio * majority_distance)
            eps = 32 * np.finfo(float).eps * max(geometry.radius, np.finfo(float).tiny)
            candidate, path = None, None
            if geometry.dimension >= 2 and radius > 0:
                candidate = self._attempt(
                    tangent_candidate,
                    anchor,
                    points,
                    geometry,
                    rng,
                    radius,
                    eps,
                    minority_tree,
                    majority_tree,
                )
                if candidate is not None:
                    path = "tangent"
            if candidate is None:
                if not self.allow_fallback:
                    raise SamplingError(
                        "Tangent sampling failed and allow_fallback=False; "
                        "inspect diagnostics_ for partial progress."
                    )
                if k > 0 and radius > 0:
                    candidate = self._attempt(
                        segment_candidate,
                        anchor,
                        points,
                        geometry,
                        rng,
                        radius,
                        eps,
                        minority_tree,
                        majority_tree,
                    )
                    if candidate is not None:
                        path = "segment"
                if candidate is None:
                    if self.fallback == "raise":
                        raise SamplingError(
                            "Unable to safely generate the requested count after "
                            "bounded attempts; fallback='raise'."
                        )
                    candidate, path = anchor.copy(), "duplicate"
            output.append(candidate)
            self.diagnostics_[path] += 1
            self.diagnostics_["generated"] += 1
            self.sample_metadata_.append(
                dict(
                    anchor_index=int(minority_indices[i]),
                    path=path,
                    dimension=geometry.dimension,
                    radius=radius,
                )
            )
        return (
            np.vstack([X, output]),
            np.concatenate([y, np.full(requested, self.minority_class_, dtype=y.dtype)]),
        )

    def _attempt(
        self, factory, anchor, points, geometry, rng, radius, eps, minority_tree, majority_tree
    ):
        for _ in range(self.max_attempts):
            self.diagnostics_["attempts"] += 1
            candidate = factory(anchor, points, geometry, rng, radius)
            if (
                not np.isfinite(candidate).all()
                or np.linalg.norm(candidate - anchor) > radius + eps
            ):
                self.diagnostics_["radius_rejections"] += 1
                continue
            # Degenerate interpolation is counted as duplicate via the final policy.
            d_min = float(minority_tree.query(candidate)[0])
            if d_min <= eps:
                continue
            if self.check_boundary:
                d_maj = float(majority_tree.query(candidate)[0])
                if not d_min + eps < self.safety_ratio * (d_maj + eps):
                    self.diagnostics_["boundary_rejections"] += 1
                    continue
            return candidate
        return None
