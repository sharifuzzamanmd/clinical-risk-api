"""Feature schema for the UCI Chronic Kidney Disease dataset (id 336).

The raw data ships under two naming conventions (UCI ARFF and the popular
Kaggle CSV). Everything is mapped onto one set of readable names here, so the
rest of the code never sees the abbreviations.
"""

from __future__ import annotations

NUMERIC_FEATURES: list[str] = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "haemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
]

# Every categorical feature in this dataset is binary. Listing the allowed
# values explicitly keeps encoding stable between training and serving.
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "red_blood_cells": ["normal", "abnormal"],
    "pus_cell": ["normal", "abnormal"],
    "pus_cell_clumps": ["notpresent", "present"],
    "bacteria": ["notpresent", "present"],
    "hypertension": ["no", "yes"],
    "diabetes_mellitus": ["no", "yes"],
    "coronary_artery_disease": ["no", "yes"],
    "appetite": ["good", "poor"],
    "pedal_edema": ["no", "yes"],
    "anemia": ["no", "yes"],
}

FEATURES: list[str] = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES)
TARGET = "ckd"

# raw column name -> readable name (covers UCI and Kaggle variants)
COLUMN_MAP: dict[str, str] = {
    "age": "age",
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "red_blood_cells",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sc": "serum_creatinine",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "haemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "wbcc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "rbcc": "red_blood_cell_count",
    "htn": "hypertension",
    "dm": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "appet": "appetite",
    "pe": "pedal_edema",
    "ane": "anemia",
    "classification": "class",
    "class": "class",
}

UNITS: dict[str, str] = {
    "age": "years",
    "blood_pressure": "mm/Hg",
    "blood_glucose_random": "mg/dL",
    "blood_urea": "mg/dL",
    "serum_creatinine": "mg/dL",
    "sodium": "mEq/L",
    "potassium": "mEq/L",
    "haemoglobin": "g/dL",
    "white_blood_cell_count": "cells/cmm",
    "red_blood_cell_count": "millions/cmm",
}
