# Changelog

## 0.1.0 — 2026-11-20

First experimental release of Adaptive Tangent-Space SMOTE (`ats-smote`, `ats_smote`).

- Weighted local PCA dimension selection with support and stability heuristics.
- Projected convex-region sampling, majority-aware radius and distance acceptance.
- Bounded retries, segment interpolation, duplicate/error policies.
- Dense binary validation, private seeded RNG, sklearn cloning and imblearn Pipeline.
- Diagnostics, per-row provenance and three research ablations.
- Tests, quickstart, repeated-CV comparisons, scarcity/dimension studies and failure figures.
- Source/wheel packaging, MIT license and reproducibility documentation.

Limits: linear approximation, small-data evaluation, possible off-support
generation, neighborhood uncertainty, averaging concentration and altered
class-prior calibration. Ablation failures are not imputed. No PyPI publication
or academic-recognition claim. See results/REPORT.md for measured gains and losses.
