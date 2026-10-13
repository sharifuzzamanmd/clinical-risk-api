import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import make  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make(n=300, seed=1)


@pytest.fixture(scope="session")
def data_csv(tmp_path_factory, raw_df):
    path = tmp_path_factory.mktemp("data") / "ckd.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, data_csv):
    from ckd_risk.train import train

    out = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    train(data_csv, out)
    return out


@pytest.fixture(scope="session")
def client(model_path):
    import os

    os.environ["MODEL_PATH"] = str(model_path)
    from fastapi.testclient import TestClient

    from ckd_risk.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sick_patient():
    return {
        "age": 62,
        "blood_pressure": 90,
        "specific_gravity": 1.010,
        "albumin": 3,
        "sugar": 1,
        "red_blood_cells": "abnormal",
        "pus_cell": "abnormal",
        "pus_cell_clumps": "present",
        "bacteria": "notpresent",
        "blood_glucose_random": 210,
        "blood_urea": 90,
        "serum_creatinine": 5.2,
        "sodium": 130,
        "potassium": 5.4,
        "haemoglobin": 9.1,
        "packed_cell_volume": 29,
        "white_blood_cell_count": 10400,
        "red_blood_cell_count": 3.4,
        "hypertension": "yes",
        "diabetes_mellitus": "yes",
        "coronary_artery_disease": "no",
        "appetite": "poor",
        "pedal_edema": "yes",
        "anemia": "yes",
    }


@pytest.fixture
def healthy_patient():
    return {
        "age": 35,
        "blood_pressure": 70,
        "specific_gravity": 1.025,
        "albumin": 0,
        "sugar": 0,
        "red_blood_cells": "normal",
        "pus_cell": "normal",
        "pus_cell_clumps": "notpresent",
        "bacteria": "notpresent",
        "blood_glucose_random": 100,
        "blood_urea": 28,
        "serum_creatinine": 0.8,
        "sodium": 142,
        "potassium": 4.2,
        "haemoglobin": 15.5,
        "packed_cell_volume": 46,
        "white_blood_cell_count": 7200,
        "red_blood_cell_count": 5.5,
        "hypertension": "no",
        "diabetes_mellitus": "no",
        "coronary_artery_disease": "no",
        "appetite": "good",
        "pedal_edema": "no",
        "anemia": "no",
    }
