"""Render tables and a paired-comparison figure from recorded CSV results."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(output):
    summary = read(output / "summary.csv")
    folds = read(output / "fold_results.csv")
    index = {(r["dataset"], r["classifier"], r["method"]): r for r in summary}
    names = sorted({r["dataset"] for r in summary})
    methods = list(dict.fromkeys(r["method"] for r in summary))
    lines = [
        "# Recorded evaluation results",
        "",
        "Five seeds (0–4) × 3 stratified folds; fixed defaults; no tuning. "
        "AP = average precision. Values are descriptive means ± fold standard deviations; "
        "the overlapping folds are not independent trials.",
        "",
        "## Average precision: all methods",
        "",
    ]
    for classifier in ["logistic", "random_forest"]:
        lines += [
            f"### {classifier}",
            "",
            "| Method | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for method in methods:
            cells = []
            for name in names:
                row = index[name, classifier, method]
                successes = int(row["successes"])
                cell = (
                    (
                        f"{float(row['average_precision_mean']):.4f} ± "
                        f"{float(row['average_precision_std']):.4f}"
                    )
                    if successes
                    else "failed"
                )
                if successes != 15:
                    cell += f" ({successes}/15)"
                cells.append(cell)
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## ATS versus SMOTE",
        "",
        "| Dataset | Model | ATS AP | SMOTE AP | Δ AP | ATS recall | ATS FPR | ATS Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    plot_labels, plot_values = [], []
    for name in names:
        for classifier in ["logistic", "random_forest"]:
            ats, smote = index[name, classifier, "ats"], index[name, classifier, "smote"]
            ap, base = float(ats["average_precision_mean"]), float(smote["average_precision_mean"])
            lines.append(
                f"| {name} | {classifier} | {ap:.4f} | {base:.4f} | {ap - base:+.4f} | "
                f"{float(ats['recall_mean']):.4f} | {float(ats['false_positive_rate_mean']):.4f} | "
                f"{float(ats['brier_mean']):.4f} |"
            )
            plot_labels.append(f"{name} / {classifier}")
            plot_values.append(ap - base)
    lines += ["", "## Failures (not imputed)", ""]
    errors = {}
    for row in folds:
        if row["status"] != "ok":
            key = (row["method"], row["dataset"], row["error"])
            errors[key] = errors.get(key, 0) + 1
    for (method, name, error), count in sorted(errors.items()):
        lines.append(f"- {method}, {name}: {count} model-evaluation rows; `{error}`")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "A positive AP difference is dataset/model specific. Negative differences are "
        "retained. Resampling changes training priors; Brier scores are uncalibrated. "
        "The safety rule uses observed training distances and does not prove class membership.",
        "",
        "No-fallback failures mean the requested count could not be filled using tangent "
        "sampling alone. Partial-success ablation averages are selection-biased and are "
        "not evidence of superiority. See `summary.csv` for success counts and all metrics.",
        "",
        "Sparse annular supports can be bridged even by an accepted candidate: limited "
        "majority observations do not identify the true boundary. Inspect the fixed "
        "seed-0/fold-0 plot and `scarcity.csv`; no favorable plot was selected by score.",
        "",
        "Simplicial SMOTE is absent because the author-linked repository was empty when "
        "checked. LoRAS uses pyloras, not an exact author-code reproduction. These are small "
        "benchmarks with one real dataset, not a comprehensive ranking or novelty claim.",
        "",
        "![Paired AP difference](ap_difference.png)",
        "",
        "![Sparse annulus](sparse_annulus.png)",
        "",
        "![Generation paths](scarcity.png)",
        "",
        "## Explicit ATS failure: missing boundary observations",
        "",
        "An adversarial sample contains 12 minority points on the unit circle and "
        "100 majority points on radius 1.7, with no observations in the true inner "
        "majority disk (radius ≤ 0.75). Default ATS with seed 0 accepts 88 candidates; "
        "87 (98.86%) enter that disk. All pass the observed-distance rule. This "
        "counterexample is separate from the five-dataset CV ranking and demonstrates "
        "why acceptance does not guarantee true class membership.",
        "",
        "![ATS failure](unseen_boundary.png)",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(9, 5))
    values = np.array(plot_values)
    ax.barh(plot_labels, values, color=np.where(values >= 0, "#237b8c", "#d36b40"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Mean paired AP difference (ATS − SMOTE)",
        title="5 seeds × 3 folds; fixed settings; descriptive differences",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "ap_difference.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(parser.parse_args().output)
