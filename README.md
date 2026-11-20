# Adaptive Tangent-Space SMOTE

[CI workflow template](docs/github-actions-ci.yml) · [Validation evidence](docs/VALIDATION.md)

An experimental oversampler for **dense, numeric, binary classification**.
Distribution: **`ats-smote`** · Python import: **`ats_smote`** · version **0.1.0**.

The research question is whether expanding generation into a locally supported
tangent space, restricting candidates using majority observations, and falling
back under uncertainty improves synthetic quality and classification. Improvement
is not guaranteed. This is a **linear approximation**, not a curvature estimator.

## Install and run

Targets Python 3.10–3.13; locally tested on 3.10.12. Uses scikit-learn 1.7.x and
imbalanced-learn 0.14.x. Runtime bounds
are in `pyproject.toml`; experiment versions are in `requirements-reproduce.txt`.
No GPU, credentials or dataset download is needed.

```bash
git clone https://github.com/hublemon/adaptive-tangent-space-smote.git
cd adaptive-tangent-space-smote
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

Install from source or a release wheel. This release is **not published to PyPI**;
`pip install ats-smote` is not the installation instruction.

```python
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ats_smote import AdaptiveTangentSpaceSMOTE

X, y = make_classification(n_samples=600, n_features=8,
                          weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
model = Pipeline([
    ("scale", StandardScaler()),
    ("sample", AdaptiveTangentSpaceSMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.named_steps["sample"].diagnostics_)
```

Fit scalers and samplers only inside each training fold. The
[imbalanced-learn leakage guide](https://imbalanced-learn.org/stable/common_pitfalls.html)
explains why resampling before splitting gives misleading evaluation results.

## Algorithm and contract

Distance-weighted local PCA selects a dimension capped by rank and `max_dim`.
Insufficient support, a small eigenvalue gap or dimension disagreement after
dropping the furthest neighbor triggers one-dimensional generation. Stable
dimensions of two or more sample a projected local convex region with Dirichlet
weights. A local radius and majority-distance contraction limit displacement.
A candidate is accepted only if its nearest minority distance is sufficiently
smaller than its nearest majority distance. Bounded retries are followed by
explicit segment interpolation, then duplication or an error.

Projection can leave the original feature-space convex hull. The safety rule is
a training-data heuristic, not a guarantee of true class membership.
`max_dim=1` uses actual anchor–neighbor segments; restrictions and RNG conventions
mean the complete sampler is not identical to SMOTE.

`fit_resample(X, y)` returns NumPy arrays with original rows first and exactly the
requested number appended, or raises `SamplingError`. Inputs are copied and
features converted to float64. An integer seed controls a private RNG.
Labels may be discrete numbers, booleans or strings; exactly two classes are
required. Balanced data with `auto` returns copies. Sparse matrices, categorical
or mixed features, complex values, NaN, infinity, multilabel and multiclass
targets are rejected. Integer categorical codes are not detected automatically.
Scale features within a Pipeline; extreme magnitudes can exceed floating-point
arithmetic. Outputs do not preserve pandas container types.

| Parameter | Default | Meaning |
|---|---:|---|
| `sampling_strategy` | `"auto"` | Balance, final minority/majority ratio in (0,1], or `{minority_label: final_count}` |
| `k_neighbors` | 8 | Maximum neighbors, reduced when necessary |
| `max_dim` | 3 | Dimension cap; 1 forces segments |
| `variance_threshold` | 0.9 | Weighted explained-variance threshold |
| `safety_ratio` | 0.8 | Nearest minority/majority distance ratio |
| `max_attempts` | 20 | Attempts per stage and requested row |
| `fallback` | `"duplicate"` | Final policy: `"duplicate"` or `"raise"` |
| `random_state` | `None` | Nonnegative integer seed or private nondeterministic RNG |

Research switches `adapt_dimension`, `check_boundary`, `allow_fallback` default
to True; see [the evaluation protocol](experiments/README.md).
`diagnostics_` exposes counts, attempts, rejections, dimensions and uncertainty
reasons. `sample_metadata_` records each appended row's original anchor index,
path, selected dimension and radius. `local_dimensions_` contains local estimates.
Duplicate fallbacks preserve observations **without claiming they pass the safety
test**. A single minority observation can only be duplicated; `fallback="raise"`
reports failure instead. Inspect duplicate rates when assessing diversity.

See [the mathematical specification and code review guide](docs/ALGORITHM.md).

## Reproduce validation and experiments

```bash
python -m pip install -r requirements-reproduce.txt
python -m pip install -e '.[dev,experiments]'
python -m pytest --cov=ats_smote --cov-report=term-missing
ruff check .
python examples/quickstart.py
python -m experiments.run_benchmarks --output results
python -m experiments.geometry_study --output results
python -m experiments.report --output results
python -m build
python -m twine check dist/*
```

[Recorded results](results/REPORT.md) include all methods, paired fold results,
failures, geometry, scarcity sensitivity and figures. Five seeds × three folds
cover five datasets and two classifiers. Comparators: no resampling, class
weighting, random oversampling, SMOTE, BorderlineSMOTE, ADASYN, Geometric SMOTE,
LoRAS and three ATS ablations. Defaults were fixed before evaluation, without
test-fold tuning. Protocol details explain omitted Simplicial SMOTE and the
distinction between true-support intrusion and the noisy-moons proxy.

## Limits and provenance

Sparse or disconnected neighborhoods can produce incorrect tangent estimates
and bridge holes. Boundary checks cannot identify an unobserved true boundary.
Barycentric averaging may concentrate points near the local mean. Radius
contraction is isotropic. Oversampling changes training priors and may worsen
calibration or false positives. This is a small-data study with one public real
dataset, not a comprehensive ranking or evidence of universal superiority.

Multiple-neighbor sampling is established prior work, including
[Simplicial SMOTE](https://arxiv.org/abs/2503.03418),
[Geometric SMOTE](https://arxiv.org/abs/1709.07377), and
[LoRAS](https://doi.org/10.1007/s10994-020-05913-4).
This implementation studies dimension selection, boundary checks and fallback;
it is not an implementation of those papers or a claim of proven novelty.

Synthetic generators are included. The real dataset is
[Wisconsin Diagnostic Breast Cancer, UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
by W. Wolberg, O. Mangasarian, N. Street and W. Street (1993),
[DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), CC BY 4.0.
It is loaded through scikit-learn; malignant is relabeled 1 and original class
counts retained. No raw data is committed. This is a software benchmark,
not clinical validation. DHS data is not used.

Original code: [MIT](LICENSE). Third-party terms: [attributions](docs/ATTRIBUTION.md).
Project idea, requirements and ownership: **hublemon**. Initial code, tests and
experiments were prepared with automated development assistance; maintainer
review and academic contribution disclosure remain separate from Git authorship.
Departmental Open S/W recognition has not been verified.
See [release notes](CHANGELOG.md) and [validation evidence](docs/VALIDATION.md).
