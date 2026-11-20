# Sources and third-party terms

Original implementation and synthetic generators: MIT, copyright 2026 hublemon.
No third-party algorithm source was copied or vendored.

- Chawla et al., SMOTE (2002): https://arxiv.org/abs/1106.1813.
- Kachan, Savchenko and Gusev, Simplicial SMOTE (2025): https://arxiv.org/abs/2503.03418.
  Higher-dimensional barycentric sampling is prior art. The paper's code link
  https://github.com/oleg-kachan/simplicial-smote-kdd25 was empty on 2026-11-20.
- Douzas and Bacao, Geometric SMOTE: https://arxiv.org/abs/1709.07377.
  Experiment code: https://github.com/georgedouzas/imbalanced-learn-extra,
  version 0.2.10, MIT; sampling from geometrically constrained regions.
- Bej et al., LoRAS (2021): https://doi.org/10.1007/s10994-020-05913-4.
  Experiment code: https://github.com/zoj613/pyloras, 0.1.0b6,
  BSD-3-Clause, Zolisa Bleki; third-party implementation using noisy affine
  combinations and t-SNE neighborhoods. Author code inspected at
  https://github.com/narek-davtyan/LoRAS but not used in the benchmark.
- scikit-learn (BSD-3-Clause), imbalanced-learn (MIT), NumPy and SciPy
  (BSD-3-Clause), Matplotlib (Matplotlib license): dependencies retain their terms.
- Wisconsin Diagnostic Breast Cancer: Wolberg, Mangasarian, Street and Street
  (1993), UCI, https://doi.org/10.24432/C5DW2B, CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/.
  Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.
  Loaded through scikit-learn; malignant is recoded to 1, scaling is fitted per
  training fold. Raw rows are not redistributed here.

The supplied PDF and handoff provided motivating questions and the initial
contract. They are not research results, novelty evidence, or proof of academic
recognition. Informal scores or predicted performance in them are not repeated
as factual claims in this release.
