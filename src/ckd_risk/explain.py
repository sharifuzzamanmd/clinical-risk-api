"""Per-prediction explanations with SHAP.

The explainer works on the preprocessed feature matrix. Because the
preprocessor emits exactly one column per input feature, each SHAP value maps
back to one clinical variable. Values are in probability units: a value of
+0.12 means that feature pushed this patient's CKD probability up by 12
percentage points relative to the background population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .features import FEATURES


@dataclass
class Contribution:
    feature: str
    value: object
    shap_value: float


class RiskExplainer:
    def __init__(self, pipeline: Pipeline, background: np.ndarray, max_evals: int = 2 * len(FEATURES) + 1):
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.max_evals = max(max_evals, 2 * len(FEATURES) + 1)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        self._explainer = shap.PermutationExplainer(self._predict_positive, masker, seed=0)
        self.base_value = float(self._predict_positive(background).mean())

    def _predict_positive(self, z: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(z)[:, 1]

    def explain(self, x: pd.DataFrame, top_k: int | None = None) -> list[list[Contribution]]:
        """Return contributions per row, sorted by absolute impact."""
        z = self.preprocess.transform(x[FEATURES])
        result = self._explainer(z, max_evals=self.max_evals, silent=True)
        out: list[list[Contribution]] = []
        for i in range(len(x)):
            row = [
                Contribution(feature=f, value=_to_native(x.iloc[i][f]), shap_value=float(v))
                for f, v in zip(FEATURES, result.values[i], strict=True)
            ]
            row.sort(key=lambda c: abs(c.shap_value), reverse=True)
            out.append(row[:top_k] if top_k else row)
        return out


def _to_native(v: object) -> object:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v
