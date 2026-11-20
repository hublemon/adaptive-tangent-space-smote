"""Run from the repository root: python examples/quickstart.py."""

from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ats_smote import AdaptiveTangentSpaceSMOTE


def main():
    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.9, 0.1], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("sample", AdaptiveTangentSpaceSMOTE(random_state=42)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    print(f"Average precision: {average_precision_score(y_test, score):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, model.predict(X_test)):.4f}")
    print("Sampler diagnostics:", model.named_steps["sample"].diagnostics_)
    print(f"Training rows: {len(X_train)}; untouched test rows: {len(X_test)}")


if __name__ == "__main__":
    main()
