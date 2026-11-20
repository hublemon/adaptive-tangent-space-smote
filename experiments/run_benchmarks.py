"""python -m experiments.run_benchmarks --output results

Five seeds x three stratified folds, with shared splits and training-only scaling.
Defaults and failure policies are fixed before evaluation. No model tuning.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ats_smote import AdaptiveTangentSpaceSMOTE

from .datasets import datasets, support_metrics


def samplers(seed, extras=True):
    result = {
        "none": None,
        "class_weight": None,
        "random_over": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed),
        "borderline_smote": BorderlineSMOTE(random_state=seed),
        "adasyn": ADASYN(random_state=seed),
        "ats": AdaptiveTangentSpaceSMOTE(random_state=seed),
        "ats_fixed_dim": AdaptiveTangentSpaceSMOTE(random_state=seed, adapt_dimension=False),
        "ats_no_boundary": AdaptiveTangentSpaceSMOTE(random_state=seed, check_boundary=False),
        "ats_no_fallback": AdaptiveTangentSpaceSMOTE(random_state=seed, allow_fallback=False),
    }
    if extras:
        from imblearn_extra.gsmote import GeometricSMOTE
        from pyloras import LORAS

        result["geometric_smote"] = GeometricSMOTE(random_state=seed)
        result["loras"] = LORAS(random_state=seed)
    return result


def classifiers(seed, weighted=False):
    weight = "balanced" if weighted else None
    return {
        "logistic": LogisticRegression(max_iter=1000, C=1, class_weight=weight, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight=weight, random_state=seed, n_jobs=1
        ),
    }


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    metrics = [
        "average_precision",
        "recall",
        "balanced_accuracy",
        "false_positive_rate",
        "brier",
        "intrusion_rate",
        "support_distance",
        "unique_fraction",
        "spread_ratio",
        "neighbor_majority_fraction",
        "tangent_fraction",
        "segment_fraction",
        "duplicate_fraction",
        "boundary_rejection_fraction",
    ]
    for row in rows:
        key = (row["dataset"], row["classifier"], row["method"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        record = dict(zip(["dataset", "classifier", "method"], key))
        successful = [r for r in group if r["status"] == "ok"]
        record.update(successes=len(successful), failures=len(group) - len(successful))
        for metric in metrics:
            values = [r[metric] for r in successful if metric in r]
            if values:
                record[metric + "_mean"] = float(np.mean(values))
                record[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0
        output.append(record)
    return output


def paired_differences(rows):
    index = {(r["dataset"], r["classifier"], r["seed"], r["fold"], r["method"]): r for r in rows}
    differences = []
    for row in rows:
        if row["method"] != "ats":
            continue
        base = index[(*(row[k] for k in ["dataset", "classifier", "seed", "fold"]), "smote")]
        if row["status"] == base["status"] == "ok":
            record = {k: row[k] for k in ["dataset", "classifier", "seed", "fold", "split_id"]}
            for metric in [
                "average_precision",
                "recall",
                "balanced_accuracy",
                "false_positive_rate",
            ]:
                record[metric + "_delta"] = row[metric] - base[metric]
            differences.append(record)
    return differences


def run(output, seeds, extras=True):
    output.mkdir(parents=True, exist_ok=True)
    rows, splits, descriptions = [], [], []
    started = perf_counter()
    for name, X, y in datasets():
        descriptions.append(
            dict(
                name=name,
                rows=len(X),
                features=X.shape[1],
                minority=int(y.sum()),
                majority=int((y == 0).sum()),
            )
        )
        for seed in seeds:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(cv.split(X, y)):
                split_id = hashlib.sha256(
                    np.r_[train, -1, test].astype("<i8").tobytes()
                ).hexdigest()[:16]
                splits.append(
                    dict(
                        dataset=name,
                        seed=seed,
                        fold=fold,
                        split_id=split_id,
                        train=train.tolist(),
                        test=test.tolist(),
                    )
                )
                scaler = StandardScaler().fit(X[train])
                X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])
                y_train, y_test = y[train], y[test]
                for method, sampler in samplers(seed, extras).items():
                    common = dict(
                        dataset=name,
                        seed=seed,
                        fold=fold,
                        method=method,
                        split_id=split_id,
                        n_train=len(train),
                        n_test=len(test),
                        train_minority=int(y_train.sum()),
                        status="ok",
                        error="",
                    )
                    try:
                        t = perf_counter()
                        Xr, yr = (
                            (X_train, y_train)
                            if sampler is None
                            else sampler.fit_resample(X_train, y_train)
                        )
                        common["resample_seconds"] = perf_counter() - t
                        common["generated"] = len(Xr) - len(X_train)
                        if common["generated"]:
                            synthetic = Xr[len(X_train) :]
                            common.update(
                                support_metrics(name, scaler.inverse_transform(synthetic))
                            )
                            common["unique_fraction"] = len(np.unique(synthetic, axis=0)) / len(
                                synthetic
                            )
                            original_spread = np.var(X_train[y_train == 1], axis=0).sum()
                            common["spread_ratio"] = float(
                                np.var(synthetic, axis=0).sum() / original_spread
                            )
                            neighbors = cKDTree(X_train).query(synthetic, k=7)[1]
                            common["neighbor_majority_fraction"] = float(
                                np.mean(y_train[neighbors] == 0)
                            )
                        if hasattr(sampler, "diagnostics_"):
                            d = sampler.diagnostics_
                            common["dimensions"] = json.dumps(d["dimension_counts"], sort_keys=True)
                            for path in ["tangent", "segment", "duplicate"]:
                                common[path + "_fraction"] = d[path] / max(1, d["generated"])
                            common["boundary_rejection_fraction"] = d["boundary_rejections"] / max(
                                1, d["attempts"]
                            )
                    except (ValueError, RuntimeError) as exc:
                        common.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    for label, classifier in classifiers(seed, method == "class_weight").items():
                        record = dict(common, classifier=label)
                        if common["status"] == "ok":
                            classifier.fit(Xr, yr)
                            probabilities = classifier.predict_proba(X_test)[:, 1]
                            prediction = classifier.predict(X_test)
                            record.update(
                                average_precision=float(
                                    average_precision_score(y_test, probabilities)
                                ),
                                recall=float(recall_score(y_test, prediction)),
                                balanced_accuracy=float(
                                    balanced_accuracy_score(y_test, prediction)
                                ),
                                false_positive_rate=float(np.mean(prediction[y_test == 0] == 1)),
                                brier=float(brier_score_loss(y_test, probabilities)),
                            )
                        rows.append(record)
            print(f"{name}, seed={seed}: {len(rows)} paired model evaluations", flush=True)
            write_csv(output / "fold_results.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    write_csv(output / "paired_vs_smote.csv", paired_differences(rows))
    (output / "splits.json").write_text(json.dumps(splits, separators=(",", ":")) + "\n")
    packages = ["ats-smote", "numpy", "scipy", "scikit-learn", "imbalanced-learn", "matplotlib"]
    if extras:
        packages += ["imbalanced-learn-extra", "pyloras"]
    manifest = dict(
        seeds=seeds,
        folds=3,
        dataset_seed=2026,
        python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in packages},
        datasets=descriptions,
        elapsed_seconds=perf_counter() - started,
        parameters={name: repr(value) for name, value in samplers(0, extras).items()},
        classifiers={name: repr(value) for name, value in classifiers(0).items()},
        threads=1,
        tuning="none",
        evaluation="repeated stratified cross-validation",
        caveat="Folds and seeds overlap; no independent-sample significance claim.",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--without-extras", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.seeds, not args.without_extras)


if __name__ == "__main__":
    main()
