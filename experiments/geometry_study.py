"""Geometry, sample-scarcity sensitivity, and an intentionally difficult example."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ats_smote import AdaptiveTangentSpaceSMOTE

from .datasets import annulus, support_metrics, unseen_inner_boundary
from .run_benchmarks import write_csv


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    X_bad, y_bad = unseen_inner_boundary()
    bad_sampler = AdaptiveTangentSpaceSMOTE(random_state=0)
    bad_result, _ = bad_sampler.fit_resample(X_bad, y_bad)
    synthetic_bad = bad_result[len(X_bad) :]
    bad_metrics = support_metrics("annulus", synthetic_bad)
    (output / "unseen_boundary.json").write_text(
        json.dumps(
            {
                "purpose": "Adversarial observations missing the true inner majority support.",
                "seed": 0,
                "metrics": bad_metrics,
                "diagnostics": bad_sampler.diagnostics_,
            },
            indent=2,
        )
        + "\n"
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_patch(
        plt.Circle((0, 0), 0.75, color="#aaa", alpha=0.25, label="Unobserved true majority support")
    )
    ax.scatter(*X_bad[y_bad == 0].T, s=12, c="#888", label="Observed majority")
    ax.scatter(*X_bad[y_bad == 1].T, s=40, marker="x", c="#145a80", label="Observed minority")
    ax.scatter(*synthetic_bad.T, s=12, c="#d36b40", label="Accepted ATS candidates")
    for radius in [0.9, 1.1]:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#145a80", lw=0.8))
    ax.set(
        aspect="equal",
        title="ATS failure: unobserved inner boundary\n"
        f"True-majority intrusion: {bad_metrics['intrusion_rate']:.1%}",
    )
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "unseen_boundary.png", dpi=150)
    plt.close(fig)
    sensitivity = []
    for n_minority in [1, 2, 4, 8, 12, 30, 90]:
        X, y = annulus(n_minority=n_minority)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        for seed in range(5):
            sampler = AdaptiveTangentSpaceSMOTE(random_state=seed)
            Xr, _ = sampler.fit_resample(Xs, y)
            d = sampler.diagnostics_
            sensitivity.append(
                dict(
                    n_minority=n_minority,
                    seed=seed,
                    **d,
                    **support_metrics("annulus", scaler.inverse_transform(Xr[len(X) :])),
                )
            )
    write_csv(output / "scarcity.csv", sensitivity)

    # Controlled noiseless supports with known intrinsic dimensions.
    dimension_rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for kind, points, true_dim in [
            ("line", np.c_[np.linspace(-1, 1, 100), np.zeros((100, 2))], 1),
            ("plane", np.c_[rng.uniform(-1, 1, (100, 2)), np.zeros(100)], 2),
        ]:
            other = rng.normal(5, 1, (200, 3))
            sampler = AdaptiveTangentSpaceSMOTE(random_state=seed)
            sampler.fit_resample(np.vstack([points, other]), np.r_[np.ones(100), np.zeros(200)])
            dimension_rows.append(
                dict(
                    support=kind,
                    seed=seed,
                    true_dimension=true_dim,
                    selected_dimension_accuracy=float(
                        np.mean(sampler.local_dimensions_ == true_dim)
                    ),
                    dimension_counts=json.dumps(sampler.diagnostics_["dimension_counts"]),
                )
            )
    write_csv(output / "dimension_recovery.csv", dimension_rows)

    X, y = annulus(n_minority=12)
    train, _ = next(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    X, y = X[train], y[train]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    methods = {
        "SMOTE": SMOTE(random_state=0),
        "ATS": AdaptiveTangentSpaceSMOTE(random_state=0),
        "ATS without boundary": AdaptiveTangentSpaceSMOTE(random_state=0, check_boundary=False),
    }
    for ax, (name, sampler) in zip(axes, methods.items()):
        result, _ = sampler.fit_resample(Xs, y)
        generated = scaler.inverse_transform(result[len(X) :])
        metrics = support_metrics("annulus", generated)
        ax.scatter(*X[y == 0].T, s=7, c="#b0b8c0", alpha=0.5, label="Majority")
        ax.scatter(*generated.T, s=8, c="#e36c35", alpha=0.5, label="Generated")
        ax.scatter(*X[y == 1].T, s=35, c="#145a80", marker="x", label="Minority train")
        for radius in [0.9, 1.1]:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#145a80", lw=0.8))
        ax.set(
            title=f"{name}\nMajority-support intrusion: {metrics['intrusion_rate']:.1%}",
            xlim=(-1.9, 1.9),
            ylim=(-1.9, 1.9),
            aspect="equal",
        )
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Sparse annulus: 8 minority training rows; seed 0, fold 0")
    fig.tight_layout()
    fig.savefig(output / "sparse_annulus.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = sorted(set(r["n_minority"] for r in sensitivity))
    for key, label in [("tangent", "Tangent"), ("segment", "Segment"), ("duplicate", "Duplicate")]:
        fractions = [
            np.mean([r[key] / r["generated"] for r in sensitivity if r["n_minority"] == n])
            for n in counts
        ]
        ax.plot(counts, fractions, marker="o", label=label)
    ax.set(
        xscale="log",
        xlabel="Observed minority rows",
        ylabel="Fraction of appended rows",
        ylim=(-0.03, 1.03),
        title="ATS generation paths on an annular band (5 seeds)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "scarcity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
