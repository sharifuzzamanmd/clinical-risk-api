import json

import joblib

from ckd_risk.data import load
from ckd_risk.features import FEATURES
from ckd_risk.model import MODEL_NAMES, build_pipeline


def test_every_pipeline_fits_and_predicts(data_csv):
    X, y = load(data_csv)
    for name in MODEL_NAMES:
        proba = build_pipeline(name).fit(X, y).predict_proba(X.head(5))
        assert proba.shape == (5, 2)


def test_preprocessor_keeps_one_column_per_feature(data_csv):
    X, y = load(data_csv)
    pre = build_pipeline("stacking").named_steps["preprocess"].fit(X)
    assert list(pre.get_feature_names_out()) == FEATURES


def test_training_writes_bundle_and_metrics(model_path):
    bundle = joblib.load(model_path)
    assert {"pipeline", "background", "metrics", "version", "trained_at"} <= bundle.keys()
    metrics = json.loads((model_path.parent / "metrics.json").read_text())
    assert set(metrics["cv_5fold_train"]) == set(MODEL_NAMES)
    # synthetic data is separable, so a working pipeline must score well
    assert metrics["test"]["roc_auc"] > 0.9
