FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
# libgomp1: OpenMP runtime used by ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
# a normal user, not root
RUN useradd --create-home --uid 1000 app
COPY bangla_common.py app.py labels.json ./
# the three trained models (ONNX) and the MLflow store: both are produced by train.py, so run it before docker build
COPY models ./models
COPY --chown=app:app artifacts ./artifacts
RUN chown app:app /app
USER app
VOLUME /app/artifacts/mlflow
EXPOSE 8501 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
