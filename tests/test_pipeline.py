import numpy as np
from imblearn.pipeline import Pipeline
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from ats_smote import AdaptiveTangentSpaceSMOTE


class RecordingSampler(AdaptiveTangentSpaceSMOTE):
    def fit_resample(self, X, y):
        output = super().fit_resample(X, y)
        self.received_ = np.array(X, copy=True)
        self.received_y_ = np.array(y, copy=True)
        return output


def test_pipeline_cross_validation_and_no_test_row_leakage():
    X, y = make_classification(n_samples=180, n_features=6, weights=[0.8, 0.2], random_state=23)
    train, holdout = train_test_split(np.arange(len(y)), stratify=y, random_state=7)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("sample", RecordingSampler(random_state=3)),
            ("model", LogisticRegression(max_iter=500, random_state=3)),
        ]
    )
    cv = StratifiedKFold(3, shuffle=True, random_state=19)
    results = cross_validate(
        pipeline,
        X[train],
        y[train],
        cv=cv,
        return_estimator=True,
        scoring="average_precision",
        error_score="raise",
    )
    assert np.isfinite(results["test_score"]).all()
    for (fit, valid), model in zip(cv.split(X[train], y[train]), results["estimator"]):
        scaler, sampler = model.named_steps["scale"], model.named_steps["sample"]
        assert sampler.n_samples_in_ == len(fit)
        assert_allclose(scaler.mean_, X[train[fit]].mean(axis=0))
        assert_allclose(scaler.inverse_transform(sampler.received_), X[train[fit]])
        assert_array_equal(sampler.received_y_, y[train[fit]])
        original = sampler.received_.copy()
        assert len(model.predict(X[holdout])) == len(holdout)
        assert_array_equal(sampler.received_, original)
        assert not set(train[fit]) & set(np.r_[train[valid], holdout])
