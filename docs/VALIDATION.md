# Validation evidence — 2026-11-20

Local execution: Linux; Python 3.10.12 for experiments. Numerical dependencies are pinned in
requirements-reproduce.txt; the complete experiment environment is in
results/environment.json. Tests and commands were actually executed.

## Package and behavior

- Editable installation into both the development environment and a newly
  created, isolated virtual environment passed. The clean install used
  `pip install -c requirements-reproduce.txt -e '.[dev,experiments]'` with build isolation.
- `python -m pytest --cov=ats_smote --cov-report=term-missing`: **54 passed**, runtime
  package statement coverage **98% (224/228)**. Tests cover exact row counts,
  input preservation, label handling, seed/global-RNG behavior, clone/set_params,
  tangent/segment geometry, safety/radius, small/duplicate samples, bounded
  rejection, invalid inputs, repeated fits, and Pipeline/CV leakage prevention.
- `ruff check .`: passed. `pip check`: no broken requirements.
- `python examples/quickstart.py`: AP **0.5681**, balanced accuracy **0.7937**;
  420 training and 180 untouched test rows. Added rows: 248 tangent, 80 segment,
  0 duplicate. 1,305 boundary rejections out of 1,633 attempts.
- The complete Python block in README.md was executed successfully.
- Additional isolated Python **3.11.15, 3.12.13 and 3.13.14** environments each
  passed all **54 tests** and quickstart. All four Python versions produced the
  same displayed quickstart metrics and path/rejection counts with the pinned
  numerical libraries. This is local compatibility evidence, not a remote CI run.
- Source distribution and wheel build, `twine check`, archive-content checks
  and installation from the wheel passed. The installed module was verified
  to load from site-packages rather than the development source tree.

## Experiments

- Fixed seeds 0–4 × 3 stratified folds × 5 datasets × 12 methods × 2 classifiers:
  **1,800 recorded model evaluations**, 1,656 successful and 144 explicit failures.
- All **150 ATS evaluations** succeeded. Removing fallback failed on 71 of 75
  splits (142 model rows). ADASYN failed on one moons split (2 model rows).
- **360** seed-0 smoke evaluations matched the corresponding full-run rows
  exactly, excluding runtime. Split identities and 75 disjoint train/test
  partitions were checked. Repeated CV folds across seeds are not independent.
- 35 scarcity experiments (7 minority sample counts × 5 seeds) and 10 controlled
  intrinsic-dimension experiments were executed. Line recovery was 100%; plane
  recovery averaged 98.2%. These noiseless checks do not establish accuracy on
  noisy or unknown manifolds.
- A separate adversarial missing-boundary case accepted 88 candidates, of which
  87 (98.86%) entered the unobserved true majority region. A test preserves this
  counterexample to any claim that empirical safety proves true support.

The largest mean ATS-vs-SMOTE AP gain in this suite is sparse-annulus Random
Forest: 0.4288 versus 0.3401. Breast-cancer results are slightly worse for ATS:
0.9888 versus 0.9893 (Random Forest), 0.9940 versus 0.9941 (logistic regression).
No significance or overall superiority claim is made. Full paired results,
ablation tables, false positives, Brier scores and geometry are in results/REPORT.md
and the accompanying CSV files.

## Issues found and resolved

The initial environment lacked a sufficiently recent setuptools and wheel;
installing the declared build backend resolved editable installation. An initial
ablation test incorrectly required boundary checks to change output for widely
separated classes; it now exercises a nearby-class case and passes. Source
distribution inspection found examples/experiments were omitted by default;
MANIFEST.in now includes the reproducibility materials. No sampler defaults were
changed in response to benchmark scores.

GitHub Actions was not activated: the authenticated hublemon HTTPS credential
lacks the workflow scope, and the available SSH key is a deploy key for another
repository. The prepared Python 3.10–3.13 workflow is preserved as
`docs/github-actions-ci.yml`, outside the active workflows directory. No remote
CI success is claimed. To activate it after renewing the appropriate credential:

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows
cp docs/github-actions-ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m 'Enable Python compatibility CI'
git push
```

This scope limitation affects workflow activation; package code, local test
results, experiments and release archives can be published with the existing
repository permission.
