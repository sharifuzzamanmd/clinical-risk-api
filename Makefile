.PHONY: install data train train-mlflow serve test lint format docker-build docker-run

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py

train: data
	python -m ckd_risk.train --data data/kidney_disease.csv --out artifacts/model.joblib

train-mlflow: data
	python -m ckd_risk.train --data data/kidney_disease.csv --out artifacts/model.joblib --mlflow

serve:
	uvicorn ckd_risk.api:app --reload

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

docker-build:
	docker build -t clinical-risk-api:latest .

docker-run:
	docker run --rm -p 8000:8000 clinical-risk-api:latest
