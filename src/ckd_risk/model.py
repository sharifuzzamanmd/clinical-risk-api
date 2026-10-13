"""Model definitions.

All preprocessing lives inside one scikit-learn ``Pipeline``. Imputation and
scaling are fitted on training folds only, so cross-validation scores are not
inflated by information from the test data.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=list(CATEGORICAL_FEATURES.values()))),
        ]
    )
    # Output columns keep the input order: numeric first, then categorical.
    # One output column per input feature makes SHAP values map 1:1 to features.
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )


def base_estimators() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "naive_bayes": GaussianNB(),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_stacking() -> StackingClassifier:
    """Stacking ensemble from the AIAI 2025 paper: LR + NB + GB, meta-learner LR."""
    return StackingClassifier(
        estimators=list(base_estimators().items()),
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=5,
    )


def build_pipeline(name: str = "stacking") -> Pipeline:
    if name == "stacking":
        clf = build_stacking()
    else:
        clf = base_estimators()[name]
    return Pipeline([("preprocess", build_preprocessor()), ("model", clf)])


MODEL_NAMES = ["logistic_regression", "naive_bayes", "gradient_boosting", "stacking"]
