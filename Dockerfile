# Dockerfile
FROM python:3.10-slim

# System deps (optional but good for pandas/numpy perf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run training pipeline at build time (for demo)
RUN python -m src.data.generate_synthetic_data && \
    python -m src.autoencoder.train && \
    python -m src.twotower.train && \
    python -m src.gnn.train && \
    python -m src.lightgbm_model.train_lgbm && \
    python -m src.analytics.prepare_simulation_data

EXPOSE 8000

CMD uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT

