import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone

from ats_smote import AdaptiveTangentSpaceSMOTE as ATS
from ats_smote import SamplingError
from ats_smote.geometry import local_geometry


@pytest.fixture
def data():
    rng = np.random.default_rng(51)
    return (
        np.vstack([rng.normal(size=(25, 3)), rng.normal(6, 1, size=(65, 3))]),
        np.array(["rare"] * 25 + ["common"] * 65),
    )


def test_count_preservation_seed_and_global_rng(data):
    X, y = data
    before_x, before_y = X.copy(), y.copy()
    state = np.random.get_state()
    a, b = ATS(random_state=10), ATS(random_state=10)
    Xa, ya = a.fit_resample(X, y)
    Xb, yb = b.fit_resample(X, y)
    assert_array_equal(Xa, Xb)
    assert_array_equal(ya, yb)
    assert_array_equal(X, before_x)
    assert_array_equal(y, before_y)
    assert_array_equal(Xa[: len(X)], X)
    assert_array_equal(ya[: len(y)], y)
    assert_array_equal(np.unique(ya, return_counts=True)[1], [65, 65])
    assert not np.array_equal(Xa, ATS(random_state=11).fit_resample(X, y)[0])
    assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]
    d = a.diagnostics_
    assert d["generated"] == d["tangent"] + d["segment"] + d["duplicate"] == 40
    assert d["tangent"] > 0
    assert sum(d["dimension_counts"].values()) == 25


def test_safety_and_radius_of_every_synthesized_row(data):
    X, y = data
    sampler = ATS(random_state=41)
    new, _ = sampler.fit_resample(X, y)
    for candidate, meta in zip(new[len(X) :], sampler.sample_metadata_):
        if meta["path"] == "duplicate":
            continue
        d_min = np.linalg.norm(X[y == "rare"] - candidate, axis=1).min()
        d_maj = np.linalg.norm(X[y != "rare"] - candidate, axis=1).min()
        assert d_min < sampler.safety_ratio * d_maj
        assert np.linalg.norm(candidate - X[meta["anchor_index"]]) <= meta["radius"] + 1e-12


def test_dimension_one_really_interpolates(data):
    X, y = data
    sampler = ATS(max_dim=1, random_state=9)
    new, _ = sampler.fit_resample(X, y)
    assert sampler.diagnostics_["tangent"] == 0
    assert sampler.diagnostics_["segment"] == 40
    for candidate, meta in zip(new[len(X) :], sampler.sample_metadata_):
        anchor = X[meta["anchor_index"]]
        deltas = X[y == "rare"] - anchor
        delta = candidate - anchor
        denominators = np.sum(deltas**2, axis=1)
        nz = denominators > 0
        coefficients = (deltas[nz] @ delta) / denominators[nz]
        residuals = np.linalg.norm(delta - coefficients[:, None] * deltas[nz], axis=1)
        assert np.any((residuals < 1e-10) & (coefficients >= 0) & (coefficients <= 1))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_few_minority_rows(n):
    X = np.vstack([np.arange(n)[:, None], np.arange(20, 27)[:, None]])
    y = np.r_[np.ones(n), np.zeros(7)]
    sampler = ATS(random_state=0)
    new, labels = sampler.fit_resample(X, y)
    assert len(new) == len(labels) == 14
    assert sampler.diagnostics_["tangent"] == 0
    assert sampler.diagnostics_["duplicate" if n == 1 else "segment"] == 7 - n


def test_coincident_classes_and_zero_rank():
    X, y = np.zeros((10, 2)), np.array([1] * 3 + [0] * 7)
    sampler = ATS(max_attempts=2, random_state=0)
    sampler.fit_resample(X, y)
    assert sampler.diagnostics_["duplicate"] == 4
    assert sampler.diagnostics_["attempts"] <= 2 * 2 * 4
    with pytest.raises(SamplingError, match="bounded attempts"):
        ATS(fallback="raise").fit_resample(X, y)


def test_boundary_rejection_and_bounded_retry():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(15, 2))
    # A dense majority grid restricts candidates near each observed minority point.
    grid = np.linspace(-3, 3, 30)
    majority = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
    X, y = np.vstack([points, majority]), np.r_[np.ones(15), np.zeros(900)]
    sampler = ATS(random_state=12, max_attempts=1, sampling_strategy={1: 55})
    sampler.fit_resample(X, y)
    assert sampler.diagnostics_["boundary_rejections"] > 0
    assert sampler.diagnostics_["attempts"] <= 80
    assert sampler.diagnostics_["generated"] == 40
    assert sampler.diagnostics_["duplicate"] > 0


def test_ablation_changes_output_and_fallback_can_raise(data):
    X, y = data
    reference = ATS(random_state=1).fit_resample(X, y)[0]
    for params in ({"max_dim": 1}, {"adapt_dimension": False}):
        actual = ATS(random_state=1, **params).fit_resample(X, y)[0]
        assert not np.array_equal(actual, reference)
    near = X.copy()
    near[y == "common"] -= 5
    safe = ATS(random_state=1).fit_resample(near, y)[0]
    unchecked = ATS(random_state=1, check_boundary=False).fit_resample(near, y)[0]
    assert not np.array_equal(safe, unchecked)
    with pytest.raises(SamplingError, match="allow_fallback=False"):
        ATS(max_dim=1, allow_fallback=False).fit_resample(X, y)


def test_geometry_line_plane_and_uncertainty():
    line = np.c_[np.arange(9), np.zeros(9), np.zeros(9)]
    assert local_geometry(line, 0, np.arange(1, 9), 3, 0.9).dimension == 1
    plane = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
        ]
    )
    assert local_geometry(plane, 0, np.arange(1, 9), 3, 0.9).dimension == 2
    uncertain = local_geometry(plane[:3], 0, np.array([1, 2]), 3, 1)
    assert uncertain.dimension == 1


@pytest.mark.parametrize("strategy,expected", [("auto", 130), (0.6, 104), ({"rare": 30}, 95)])
def test_strategy_and_estimator_protocol(data, strategy, expected):
    estimator = ATS(sampling_strategy=strategy, random_state=8)
    copied = clone(estimator)
    assert copied.get_params() == estimator.get_params()
    copied.set_params(k_neighbors=4)
    X, y = data
    assert len(copied.fit_resample(X, y)[0]) == expected
    assert copied.fit(X, y) is copied
    assert copied.n_features_in_ == 3


def test_balanced_data_is_copy():
    X, y = np.arange(12).reshape(6, 2), [0, 0, 0, 1, 1, 1]
    sampler = ATS()
    result, labels = sampler.fit_resample(X, y)
    assert_array_equal(result, X)
    assert_array_equal(labels, y)
    assert not np.shares_memory(result, X)
    assert sampler.diagnostics_["generated"] == 0


def test_translation_scale_equivariance(data):
    X, y = data
    a = ATS(random_state=5).fit_resample(X, y)[0]
    b = ATS(random_state=5).fit_resample(7 * X + 20, y)[0]
    assert_allclose(7 * a + 20, b, atol=1e-10)


@pytest.mark.parametrize(
    "X,y,match",
    [
        ([["a"], ["b"]], [0, 1], "real numeric"),
        ([[1j], [2j]], [0, 1], "real numeric"),
        ([[np.nan], [1]], [0, 1], "finite"),
        ([[np.inf], [1]], [0, 1], "finite"),
        ([1, 2], [0, 1], "2-dimensional"),
        (np.empty((0, 2)), [], "nonempty"),
        ([[1], [2]], [0], "same number"),
        ([[1], [2]], [[0], [1]], "1-dimensional"),
        ([[1], [2]], [0, 0], "two classes"),
        ([[1], [2], [3]], [0, 1, 2], "two classes"),
        (csr_matrix(np.eye(2)), [0, 1], "dense"),
        ([[1], [2]], [np.nan, 1], "finite labels"),
        ([[1], [2]], [None, 1], "labels"),
    ],
)
def test_invalid_data(X, y, match):
    with pytest.raises(ValueError, match=match):
        ATS().fit_resample(X, y)


@pytest.mark.parametrize(
    "params",
    [
        {"k_neighbors": 0},
        {"k_neighbors": True},
        {"k_neighbors": 1.5},
        {"max_dim": 0},
        {"max_attempts": 0},
        {"variance_threshold": 0},
        {"variance_threshold": 1.1},
        {"safety_ratio": np.nan},
        {"safety_ratio": True},
        {"fallback": "skip"},
        {"random_state": -1},
        {"random_state": np.random.default_rng()},
        {"adapt_dimension": 1},
        {"check_boundary": "no"},
        {"allow_fallback": None},
        {"sampling_strategy": "bad"},
        {"sampling_strategy": 0.1},
        {"sampling_strategy": 1.1},
        {"sampling_strategy": True},
        {"sampling_strategy": {"common": 90}},
        {"sampling_strategy": {"rare": 30.5}},
    ],
)
def test_invalid_parameters(data, params):
    with pytest.raises(ValueError):
        ATS(**params).fit_resample(*data)


def test_repeated_fit_resets_diagnostics(data):
    sampler = ATS(random_state=3)
    sampler.fit_resample(*data)
    sampler.fit_resample(np.arange(8).reshape(4, 2), [0, 0, 1, 1])
    assert sampler.diagnostics_["generated"] == 0
    assert sampler.sample_metadata_ == []
    with pytest.raises(ValueError):
        sampler.fit_resample([[0], [1]], [0, 0])
    assert not hasattr(sampler, "diagnostics_")
