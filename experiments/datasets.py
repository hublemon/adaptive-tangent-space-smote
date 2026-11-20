"""Fixed, fully specified synthetic data and one bundled public dataset."""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.datasets import load_breast_cancer, make_moons


def annulus(n_minority=90, n_majority=450, seed=2026):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n_minority + n_majority)
    minor_r = np.sqrt(rng.uniform(0.9**2, 1.1**2, n_minority))
    n_inner = n_majority // 2
    major_r = np.r_[
        np.sqrt(rng.uniform(0, 0.75**2, n_inner)),
        np.sqrt(rng.uniform(1.25**2, 1.8**2, n_majority - n_inner)),
    ]
    radius = np.r_[minor_r, major_r]
    return (
        np.c_[radius * np.cos(angle), radius * np.sin(angle)],
        np.r_[np.ones(n_minority, dtype=int), np.zeros(n_majority, dtype=int)],
    )


def datasets():
    rng = np.random.default_rng(2026)
    X, y = make_moons(n_samples=(450, 90), noise=0.12, random_state=2026)
    yield "moons", X, y
    yield "annulus", *annulus()
    yield "annulus_sparse", *annulus(n_minority=12)
    X = np.vstack(
        [rng.uniform([-1, -1], [1, 1], (90, 2)), rng.uniform([1.3, -1], [3.3, 1], (450, 2))]
    )
    yield "flat", X, np.r_[np.ones(90, dtype=int), np.zeros(450, dtype=int)]
    data = load_breast_cancer()
    yield "breast_cancer", data.data, (data.target == 0).astype(int)


def unseen_inner_boundary():
    """Adversarial observations missing the true inner majority disk."""
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    minority = np.c_[np.cos(angle), np.sin(angle)]
    angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    majority = 1.7 * np.c_[np.cos(angle), np.sin(angle)]
    return np.vstack([minority, majority]), np.r_[np.ones(12), np.zeros(100)]


def support_metrics(name, points):
    if not len(points) or name == "breast_cancer":
        return {}
    if name.startswith("annulus"):
        radius = np.linalg.norm(points, axis=1)
        distance = np.maximum(np.maximum(0.9 - radius, radius - 1.1), 0)
        # Actual generating majority support, excluding the empty gaps.
        intrusion = (radius <= 0.75) | ((radius >= 1.25) & (radius <= 1.8))
    elif name == "flat":
        distance = np.linalg.norm(np.maximum(np.abs(points) - 1, 0), axis=1)
        intrusion = (points[:, 0] >= 1.3) & (points[:, 0] <= 3.3) & (np.abs(points[:, 1]) <= 1)
    else:
        # Moons have Gaussian noise, so the nearest noiseless arc is a proxy, not a Bayes boundary.
        theta = np.linspace(0, np.pi, 4096)
        minor = np.c_[1 - np.cos(theta), 0.5 - np.sin(theta)]
        major = np.c_[np.cos(theta), np.sin(theta)]
        distance = cKDTree(minor).query(points)[0]
        intrusion = cKDTree(major).query(points)[0] < distance
    return {
        "support_distance": float(np.mean(distance)),
        "intrusion_rate": float(np.mean(intrusion)),
    }
