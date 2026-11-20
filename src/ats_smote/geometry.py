"""Weighted local linear geometry; no curvature estimator is used."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalGeometry:
    neighbors: np.ndarray
    basis: np.ndarray
    dimension: int
    radius: float
    reason: str


def _spectrum(offsets):
    distances = np.linalg.norm(offsets, axis=1)
    positive = distances[distances > 0]
    if not len(positive):
        return np.zeros(0), np.zeros((0, offsets.shape[1]))
    scale = np.median(positive)
    weights = np.exp(-0.5 * (distances / scale) ** 2)
    weights /= weights.sum()
    center = weights @ offsets
    _, singular, axes = np.linalg.svd(
        (offsets - center) * np.sqrt(weights[:, None]), full_matrices=False
    )
    variance = singular**2
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10)) if singular[0] else 0
    return variance[:rank], axes[:rank]


def _dimension(variance, max_dim, threshold):
    if not len(variance):
        return 1
    return min(
        max_dim,
        len(variance),
        int(np.searchsorted(np.cumsum(variance) / variance.sum(), threshold)) + 1,
    )


def local_geometry(points, index, neighbors, max_dim, threshold, adapt=True):
    anchor = points[index]
    offsets = np.vstack([np.zeros(points.shape[1]), points[neighbors] - anchor])
    radius = float(np.max(np.linalg.norm(offsets, axis=1)))
    variance, axes = _spectrum(offsets)
    if not len(variance):
        return LocalGeometry(neighbors, axes, 1, radius, "zero_rank")
    dimension = _dimension(variance, max_dim, threshold) if adapt else min(max_dim, len(variance))
    reason = "selected"
    if adapt and dimension >= 2:
        # Enough distinct observations, and agreement after removing the furthest neighbor.
        reduced, _ = _spectrum(offsets[:-1])
        enough = len(np.unique(offsets, axis=0)) >= dimension + 2
        stable = _dimension(reduced, max_dim, threshold) == dimension
        gap_ok = dimension == len(variance) or variance[dimension - 1] >= 1.1 * variance[dimension]
        if not (enough and stable and gap_ok):
            dimension, reason = 1, "uncertain"
    return LocalGeometry(neighbors, axes[:dimension].T, dimension, radius, reason)


def tangent_candidate(anchor, points, geometry, rng, radius):
    offsets = np.vstack([np.zeros(anchor.shape), points[geometry.neighbors] - anchor])
    weights = rng.dirichlet(np.ones(len(offsets)))
    displacement = (weights @ (offsets @ geometry.basis)) @ geometry.basis.T
    length = np.linalg.norm(displacement)
    if length > radius:
        displacement *= radius / length
    return anchor + displacement


def segment_candidate(anchor, points, geometry, rng, radius):
    neighbor = points[rng.choice(geometry.neighbors)]
    delta = neighbor - anchor
    length = np.linalg.norm(delta)
    upper = min(1.0, radius / length) if length > 0 else 0.0
    return anchor + rng.uniform(0, upper) * delta
