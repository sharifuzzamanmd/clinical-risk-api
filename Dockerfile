# ---------- stage 1: train the model inside the build ----------
FROM python:3.12-slim AS train
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . certifi
COPY scripts ./scripts
COPY data ./data
# downloads UCI data unless data/kidney_disease.csv was already provided
RUN python scripts/download_data.py && python -m ckd_risk.train --data data/kidney_disease.csv --out artifacts/model.joblib

# ---------- stage 2: slim runtime image ----------
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/app/artifacts/model.joblib
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 appuser
COPY --from=train /app/artifacts ./artifacts
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "ckd_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
