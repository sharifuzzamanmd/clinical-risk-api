"""Train, evaluate and save the CKD risk model.

Usage:
    python -m ckd_risk.train --data data/kidney_disease.csv --out artifacts/model.joblib [--mlflow]

Steps:
1. Hold out a stratified 20% test set.
2. Compare 4 models with stratified 5-fold cross-validation on the training set.
3. Refit the stacking model on the full training set, score it once on the test set.
4. Save the pipeline plus a SHAP background sample and metrics into one joblib bundle.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import __version__
from .data import load
from .features import FEATURES
from .model import MODEL_NAMES, RANDOM_STATE, build_pipeline

CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def test_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def train(data_path: str | Path, out_path: str | Path, use_mlflow: bool = False, background_size: int = 50) -> dict:
    X, y = load(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        scores = cross_validate(build_pipeline(name), X_train, y_train, cv=cv, scoring=CV_SCORING)
        comparison[name] = {m: round(float(np.mean(scores[f"test_{m}"])), 4) for m in CV_SCORING} | {
            f"{m}_std": round(float(np.std(scores[f"test_{m}"])), 4) for m in CV_SCORING
        }
        print(f"[cv] {name:20s} " + " ".join(f"{m}={comparison[name][m]:.3f}" for m in CV_SCORING))

    pipeline = build_pipeline("stacking").fit(X_train, y_train)
    held_out = {k: round(float(v), 4) for k, v in test_metrics(y_test, pipeline.predict_proba(X_test)[:, 1]).items()}
    print("[test] stacking " + " ".join(f"{k}={v:.3f}" for k, v in held_out.items()))

    background = pipeline.named_steps["preprocess"].transform(
        X_train.sample(min(background_size, len(X_train)), random_state=RANDOM_STATE)
    )

    metrics = {
        "model": "stacking (logistic regression + naive bayes + gradient boosting -> logistic regression)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "cv_5fold_train": comparison,
        "test": held_out,
    }
    bundle = {
        "pipeline": pipeline,
        "background": background,
        "features": FEATURES,
        "metrics": metrics,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    (out_path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {out_path}")

    if use_mlflow:
        _log_mlflow(pipeline, metrics, out_path)
    return metrics


def _log_mlflow(pipeline, metrics: dict, out_path: Path) -> None:
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("ckd-risk")
    with mlflow.start_run(run_name="stacking"):
        mlflow.log_params({"model": "stacking", "cv_folds": 5, "test_size": 0.2, "random_state": RANDOM_STATE})
        for name, scores in metrics["cv_5fold_train"].items():
            mlflow.log_metrics({f"cv_{name}_{k}": v for k, v in scores.items() if not k.endswith("_std")})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics["test"].items()})
        mlflow.log_artifact(str(out_path.parent / "metrics.json"))
        mlflow.sklearn.log_model(pipeline, name="model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/kidney_disease.csv")
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics and model to MLflow")
    args = parser.parse_args()
    train(args.data, args.out, use_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
