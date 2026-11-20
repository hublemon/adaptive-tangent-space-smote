import numpy as np
from numpy.testing import assert_array_equal

from ats_smote import AdaptiveTangentSpaceSMOTE
from experiments.datasets import annulus, datasets, support_metrics, unseen_inner_boundary
from experiments.run_benchmarks import paired_differences, summarize


def test_dataset_contract_and_known_support():
    for (name, X, y), (name2, X2, y2) in zip(datasets(), datasets()):
        assert name == name2
        assert_array_equal(X, X2)
        assert_array_equal(y, y2)
        assert y.sum() < len(y) / 2
        if name in ("flat", "annulus", "annulus_sparse"):
            assert support_metrics(name, X[y == 1])["intrusion_rate"] == 0
            assert support_metrics(name, X[y == 1])["support_distance"] == 0
    X, y = annulus()
    assert support_metrics("annulus", X[y == 0])["intrusion_rate"] == 1
    assert support_metrics("annulus", np.array([[0, 0]]))["support_distance"] == 0.9


def test_failed_folds_are_not_averaged_as_success():
    shared = dict(dataset="example", classifier="logistic", seed=0, fold=0, split_id="same")
    metrics = dict(
        average_precision=0.7, recall=0.5, balanced_accuracy=0.6, false_positive_rate=0.1
    )
    rows = [
        dict(shared, method="ats", status="ok", **metrics),
        dict(shared, method="smote", status="ok", **dict(metrics, average_precision=0.6)),
        dict(shared, method="ats", status="failed", fold=1),
    ]
    ats = summarize(rows)[0]
    assert ats["successes"] == ats["failures"] == 1
    assert ats["average_precision_mean"] == 0.7
    # Pairing requires the corresponding baseline row even when ATS failed.
    rows.append(dict(shared, method="smote", status="failed", fold=1))
    differences = paired_differences(rows)
    assert len(differences) == 1
    assert abs(differences[0]["average_precision_delta"] - 0.1) < 1e-12


def test_empirical_safety_does_not_prove_true_support():
    X, y = unseen_inner_boundary()
    sampler = AdaptiveTangentSpaceSMOTE(random_state=0)
    result, _ = sampler.fit_resample(X, y)
    generated = result[len(X) :]
    assert sampler.diagnostics_["boundary_rejections"] == 0
    assert support_metrics("annulus", generated)["intrusion_rate"] > 0.9
    for point in generated:
        d_min = np.linalg.norm(X[y == 1] - point, axis=1).min()
        d_maj = np.linalg.norm(X[y == 0] - point, axis=1).min()
        assert d_min < sampler.safety_ratio * d_maj
