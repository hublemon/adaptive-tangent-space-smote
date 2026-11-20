# Algorithm and code review guide

The user-supplied handoff and implementation-idea PDF were read on 2026-11-20.
The confirmed name is Adaptive Tangent-Space SMOTE. Scope: linear tangent-space
approximation for two classes; no curvature term, neural embedding, categorical
handling or multiclass extension. Original specification documents are not
redistributed here.

## Local geometry

For anchor x_i, find up to k other minority rows with a Euclidean cKDTree.
Exclude the anchor by index, including with duplicate coordinates. Use the
anchor and its neighbors. Write o_j = x_j − x_i, h = median positive neighbor
distance, w_j proportional to exp(−||o_j||²/(2h²)), and c = Σ_j w_j o_j.
Compute the SVD of rows sqrt(w_j)(o_j − c). Numerical rank keeps singular
values larger than 1e−10 times the largest. Zero-rank neighborhoods cannot
synthesize distinct points.

Choose the smallest dimension explaining `variance_threshold` of nonzero
weighted variance, capped by rank and `max_dim`. For r ≥ 2 require at least
r+2 distinct observations, agreement of the capped dimension after removing
the furthest neighbor, and a retained/excluded eigenvalue ratio ≥ 1.1 when
an excluded nonzero eigenvalue exists. Otherwise select r=1. These are fixed
stability heuristics, not confidence bounds. `adapt_dimension=False` skips
variance and stability selection but caps supported rank at max_dim.

PCA is weighted-centered, but generation is anchored at x_i. Project offsets
onto basis U, include the zero anchor, draw a ~ Dirichlet(1,…,1), and form
v = U Σ_j a_j Uᵀo_j. This is a convex combination of projected vertices,
not uniform sampling over a general polytope. It can concentrate near the local
mean and, when dimensions are removed, leave the original feature-space hull.

## Acceptance and bounded fallback

Let R_i be the furthest selected neighbor distance. With checking enabled,
R = min(R_i, safety_ratio × d_majority(x_i)). Radially contract v into this
ball when necessary. Contraction preserves the projected convex region because
it contains the zero anchor; it is isotropic, not a directional boundary model.

Require ||x−x_i|| ≤ R+eps and
d_minority(x)+eps < safety_ratio × (d_majority(x)+eps), with eps = 32 times
float64 machine epsilon times max(R_i, smallest positive normal float).
Distances use original input rows only, never synthetic rows. Non-finite or
effectively duplicate interpolation is rejected. Translation and uniform
scaling are equivariant up to rounding and neighbor ties; feature-wise scaling
changes the method.

Select one anchor uniformly per requested row. Try at most `max_attempts`
tangent candidates. If uncertain or unsuccessful, try at most `max_attempts`
segments: choose one neighbor and λ uniformly in
[0, min(1, R/||x_j−x_i||)], then apply the same acceptance test. Finally
duplicate x_i or raise according to `fallback`. `allow_fallback=False` raises
when tangent generation is unavailable or unsuccessful. Errors return no
partial resampled array; diagnostics and metadata retain completed progress.

Candidate attempts are bounded by 2 × max_attempts × requested_count. Missing
neighbors or zero radius skip impossible stages. Duplicate fallbacks are not
claimed to pass the safety test. Diagnostic path counts reflect successful
paths, not merely requested dimensions.

## Complexity and compatibility

For m minority rows, p features and k neighbors, local SVD work is approximately
O(m k p min(k,p)). Tree queries depend on dimension and may approach brute force.
At most O(g A) candidate queries occur for g requested rows and A attempts per
stage. Geometry storage is O(m(k+p r)); output and metadata grow with g.

BaseEstimator supplies get_params, set_params, clone and tags. `fit` validates
and sets counts without generation; `fit_resample` constructs geometry and
generates rows. Use imbalanced-learn Pipeline, not standard sklearn Pipeline.
Full sklearn estimator-check compliance or all imbalanced-learn input types
are not claimed. Supported versions are in pyproject.toml.

## Review order

1. `src/ats_smote/validation.py`: inputs, parameters, requested counts.
2. `geometry.py`: weighting, SVD, stability and generation formulas.
3. `sampler.py`: original rows, trees, radius, acceptance, fallback, diagnostics.
4. `tests/`: seed/count contracts, geometric invariants, failures, CV leakage.
5. `experiments/datasets.py`: fixed distributions and support metrics.
6. `run_benchmarks.py`: common splits, train-only scaling, metrics and failures.
7. `geometry_study.py` / `report.py`: scarcity, dimensions, plots and tables.
