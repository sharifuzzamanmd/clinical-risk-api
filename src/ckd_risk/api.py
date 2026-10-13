"""FastAPI service.

Run locally:
    uvicorn ckd_risk.api:app --reload
Then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request

from .explain import RiskExplainer
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .schemas import (
    BatchRequest,
    BatchResponse,
    Explanation,
    FeatureContribution,
    ModelInfo,
    PatientFeatures,
    Prediction,
)

logger = logging.getLogger("ckd_risk")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

THRESHOLD = float(os.getenv("CKD_THRESHOLD", "0.5"))


class ModelService:
    def __init__(self, path: str | Path):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.version: str = bundle["version"]
        self.trained_at: str = bundle["trained_at"]
        self.metrics: dict = bundle["metrics"]
        self.explainer = RiskExplainer(self.pipeline, bundle["background"])

    @staticmethod
    def to_frame(patients: list[PatientFeatures]) -> pd.DataFrame:
        df = pd.DataFrame([p.model_dump() for p in patients], columns=FEATURES)
        df = df.astype({f: "float64" for f in NUMERIC_FEATURES})
        # sklearn's imputer detects NaN, not None, in text columns
        for f in CATEGORICAL_FEATURES:
            df[f] = df[f].astype(object).where(df[f].notna(), np.nan)
        return df

    def predict(self, patients: list[PatientFeatures]) -> list[Prediction]:
        x = self.to_frame(patients)
        probs = self.pipeline.predict_proba(x)[:, 1]
        return [self._prediction(float(p), x.iloc[i]) for i, p in enumerate(probs)]

    def _prediction(self, prob: float, row: pd.Series) -> Prediction:
        return Prediction(
            ckd_probability=round(prob, 4),
            prediction="ckd" if prob >= THRESHOLD else "not_ckd",
            risk_level="high" if prob >= 0.7 else "moderate" if prob >= 0.3 else "low",
            threshold=THRESHOLD,
            missing_features=[f for f in FEATURES if pd.isna(row[f])],
            model_version=self.version,
        )


state: dict[str, ModelService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    logger.info("loading model from %s", path)
    state["service"] = ModelService(path)
    yield
    state.clear()


app = FastAPI(
    title="Clinical Risk API: Chronic Kidney Disease",
    version="0.1.0",
    description=(
        "Predicts chronic kidney disease risk from routine clinical data and explains each prediction with SHAP. "
        "Research and portfolio project; **not a medical device** and not for clinical use."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def service() -> ModelService:
    if "service" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return state["service"]


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "model_loaded": "service" in state}


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
def model_info() -> ModelInfo:
    s = service()
    return ModelInfo(model_version=s.version, trained_at=s.trained_at, features=FEATURES, metrics=s.metrics)


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(patient: PatientFeatures) -> Prediction:
    return service().predict([patient])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=service().predict(req.patients))


@app.post("/explain", response_model=Explanation, tags=["prediction"])
def explain(patient: PatientFeatures, top_k: int = Query(10, ge=1, le=len(FEATURES))) -> Explanation:
    s = service()
    pred = s.predict([patient])[0]
    contributions = s.explainer.explain(s.to_frame([patient]), top_k=top_k)[0]
    return Explanation(
        **pred.model_dump(),
        base_value=round(s.explainer.base_value, 4),
        contributions=[
            FeatureContribution(
                feature=c.feature,
                value=c.value,
                imputed=c.feature in pred.missing_features,
                shap_value=round(c.shap_value, 4),
            )
            for c in contributions
        ],
    )
