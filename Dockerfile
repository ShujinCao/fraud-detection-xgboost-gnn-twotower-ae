FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data & train models on build (for demo)
RUN python -m src.data.generate_synthetic_data && \
    python -m src.autoencoder.train && \
    python -m src.twotower.train && \
    python -m src.gnn.train && \
    python -m src.xgboost_model.train_xgb

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

