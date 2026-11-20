# Evaluation protocol (fixed before the full run)

Run `python -m experiments.run_benchmarks --output results` from the repository root.
Install `pip install -e '.[experiments]'` and
`pip install imbalanced-learn-extra==0.2.10 pyloras==0.1.0b6` first.
Use `--seeds 0 --output results/smoke` for a smaller execution check.

The fixed dataset seed is 2026. Split seeds are 0–4 with 3-fold stratification.
The same split, training-only StandardScaler, classifier settings and seed are
shared by all methods. The test fold is never resampled. Sampling is cached for
the two classifiers within a split; this is equivalent to the training operations
in the documented Pipeline. No hyperparameter search or held-out-score tuning is
performed. There is no separate final holdout beyond these CV test folds.

Classifiers: LogisticRegression(C=1, max_iter=1000) and
RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=1).
Class weighting uses `balanced`; all other model settings stay equal.
External samplers use their defaults with the common seed. ATS defaults are
k=8, max_dim=3, variance_threshold=.9, safety_ratio=.8, max_attempts=20.

Three ablations use identical splits: `adapt_dimension=False` (supported rank
capped at max_dim, no variance/stability selection), `check_boundary=False`
(both majority-distance acceptance and majority-aware radius contraction off),
and `allow_fallback=False` (no segments or duplicates, explicit failure when
tangent generation cannot fill the request). All still use the local radius.
Failures are recorded as failed rows, never substituted with another sampler.
Means for methods with failures describe only successful folds and must not be
compared as if they covered every fold. ATS-vs-SMOTE differences use paired rows.

Metrics: average precision (not trapezoidal PR integration), minority recall,
balanced accuracy, majority false-positive rate, and uncalibrated Brier score.
Synthetic geometry uses the inverse-scaled generated rows, including duplicate
fallbacks. Intrusion is membership in the known majority support for annuli and
flat data. For noisy moons it is a nearest-noiseless-arc proxy, not a true boundary.
Support distance is distance to the minority annular band/rectangle or noiseless
moon arc (4096-point approximation). Unique fraction, covariance trace ratio,
7-neighbor majority fraction and ATS path/rejection fractions are also stored.

The dataset suite includes noisy moons (450:90), a 2D annular band (450:90), a
sparse annular band (450:12), separated flat rectangles (450:90), and the bundled
Wisconsin Diagnostic Breast Cancer data (357 benign:212 malignant; malignant
is mapped to label 1, without resampling the original dataset before CV).
The sparse annulus intentionally tests insufficient support and curved geometry.
ADASYN can fail on separated data with no majority neighbors; this is reported.

The splits overlap across seeds. Standard deviations are descriptive, not
standard errors; no p-values or universal superiority claims are made. Timing
is for this machine only. No DHS data is used.

Geometric SMOTE uses its author's `imbalanced-learn-extra` distribution. LoRAS
uses the third-party BSD-3-Clause `pyloras` implementation (t-SNE neighborhood
embedding, default perplexity 5), not a claim of exact author-code reproduction.
Both were installed and fit_resample-tested before inclusion. The author LoRAS
distribution `loras==0.0.22b0` has a different class-separated API, an undeclared
pandas import and process-pool behavior; it is not substituted silently.
The Simplicial SMOTE paper's linked repository
https://github.com/oleg-kachan/simplicial-smote-kdd25 returned “This repository
is empty” on 2026-11-20, so a validated implementation was unavailable. No
approximation is labeled Simplicial SMOTE.
