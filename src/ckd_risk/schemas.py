"""Request and response models for the API.

Every input is optional because real patient records often lack some lab
values. Missing values are imputed with the training-set median (numbers) or
mode (categories), the same way as during training.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNo = Optional[Literal["yes", "no"]]


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 62,
                    "blood_pressure": 80,
                    "specific_gravity": 1.010,
                    "albumin": 2,
                    "sugar": 0,
                    "red_blood_cells": "normal",
                    "pus_cell": "abnormal",
                    "pus_cell_clumps": "notpresent",
                    "bacteria": "notpresent",
                    "blood_glucose_random": 140,
                    "blood_urea": 56,
                    "serum_creatinine": 2.8,
                    "sodium": 134,
                    "potassium": 4.8,
                    "haemoglobin": 10.2,
                    "packed_cell_volume": 32,
                    "white_blood_cell_count": 9800,
                    "red_blood_cell_count": 3.9,
                    "hypertension": "yes",
                    "diabetes_mellitus": "yes",
                    "coronary_artery_disease": "no",
                    "appetite": "poor",
                    "pedal_edema": "yes",
                    "anemia": "no",
                }
            ]
        },
    )

    age: Optional[float] = Field(None, ge=0, le=120, description="years")
    blood_pressure: Optional[float] = Field(None, ge=30, le=250, description="diastolic, mm/Hg")
    specific_gravity: Optional[float] = Field(None, ge=1.000, le=1.040, description="urine specific gravity")
    albumin: Optional[float] = Field(None, ge=0, le=5, description="urine albumin, 0-5 scale")
    sugar: Optional[float] = Field(None, ge=0, le=5, description="urine sugar, 0-5 scale")
    red_blood_cells: Optional[Literal["normal", "abnormal"]] = None
    pus_cell: Optional[Literal["normal", "abnormal"]] = None
    pus_cell_clumps: Optional[Literal["present", "notpresent"]] = None
    bacteria: Optional[Literal["present", "notpresent"]] = None
    blood_glucose_random: Optional[float] = Field(None, ge=20, le=600, description="mg/dL")
    blood_urea: Optional[float] = Field(None, ge=1, le=400, description="mg/dL")
    serum_creatinine: Optional[float] = Field(None, ge=0.1, le=80, description="mg/dL")
    sodium: Optional[float] = Field(None, ge=1, le=200, description="mEq/L")
    potassium: Optional[float] = Field(None, ge=1, le=50, description="mEq/L")
    haemoglobin: Optional[float] = Field(None, ge=1, le=25, description="g/dL")
    packed_cell_volume: Optional[float] = Field(None, ge=5, le=70, description="%")
    white_blood_cell_count: Optional[float] = Field(None, ge=1000, le=30000, description="cells/cmm")
    red_blood_cell_count: Optional[float] = Field(None, ge=1, le=10, description="millions/cmm")
    hypertension: YesNo = None
    diabetes_mellitus: YesNo = None
    coronary_artery_disease: YesNo = None
    appetite: Optional[Literal["good", "poor"]] = None
    pedal_edema: YesNo = None
    anemia: YesNo = None


class Prediction(BaseModel):
    ckd_probability: float = Field(..., ge=0, le=1)
    prediction: Literal["ckd", "not_ckd"]
    risk_level: Literal["low", "moderate", "high"]
    threshold: float
    missing_features: list[str] = Field(default_factory=list, description="inputs that were imputed")
    model_version: str


class FeatureContribution(BaseModel):
    feature: str
    value: Optional[float | str] = Field(None, description="input value; null when it was missing and imputed")
    imputed: bool = False
    shap_value: float = Field(..., description="change in CKD probability caused by this feature")


class Explanation(Prediction):
    base_value: float = Field(..., description="average predicted probability over the background sample")
    contributions: list[FeatureContribution]


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_version: str
    trained_at: str
    features: list[str]
    metrics: dict
