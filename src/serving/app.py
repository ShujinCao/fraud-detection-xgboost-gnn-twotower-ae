# src/serving/app.py (additions)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.config import DATA_PROCESSED_DIR, MODELS_DIR
import lightgbm as lgb
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Fraud Risk Scoring API (Simulation)")

# ---- load model & training_table for /score_sync if you still want it ----
training_table_path = DATA_PROCESSED_DIR / "training_table.csv"
df_train = pd.read_csv(training_table_path)
LABEL_COL = "is_fraud"
EXCLUDE_COLS = ["claim_id", "claimant_id", "provider_id", LABEL_COL]
FEATURE_COLS = [c for c in df_train.columns if c not in EXCLUDE_COLS]

model_path = MODELS_DIR / "fraud_lgbm.txt"
lgb_model = lgb.Booster(model_file=str(model_path))

class ClaimFeatures(BaseModel):
    claim_id: int
    claimant_id: int
    provider_id: int
    claim_amount: float
    procedure_code: int
    days_since_last_claim: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score_sync")
def score_sync(claim: ClaimFeatures):
    row = df_train[df_train["claim_id"] == claim.claim_id]
    if row.empty:
        feature_values = [0.0] * len(FEATURE_COLS)
    else:
        feature_values = row.iloc[0][FEATURE_COLS].values

    X = pd.DataFrame([feature_values], columns=FEATURE_COLS)
    prob = float(lgb_model.predict(X)[0])
    return {"fraud_risk_score": prob}

# ---------------------- Simulation chart endpoints ------------------------

@app.get("/charts/fraud_trend")
def fraud_trend(limit: int = 500):
    df = pd.read_csv(DATA_PROCESSED_DIR / "sim_claim_timeseries.csv",
                     parse_dates=["timestamp"])
    df = df.sort_values("timestamp").tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
        "gnn_risk": df["gnn_risk"].tolist(),
    }

@app.get("/charts/anomalies")
def anomalies(limit: int = 500, threshold: float = 0.95):
    df = pd.read_csv(DATA_PROCESSED_DIR / "sim_claim_timeseries.csv",
                     parse_dates=["timestamp"])
    df = df.sort_values("timestamp").tail(limit)
    high = df[df["anomaly_score"] >= df["anomaly_score"].quantile(threshold)]
    return {
        "timestamp": high["timestamp"].astype(str).tolist(),
        "anomaly_score": high["anomaly_score"].tolist(),
        "claim_id": high["claim_id"].tolist(),
    }

@app.get("/charts/provider_risk")
def provider_risk():
    df = pd.read_csv(DATA_PROCESSED_DIR / "sim_provider_risk.csv")
    df = df.sort_values("avg_fraud", ascending=False).head(20)
    return {
        "provider_id": df["provider_id"].tolist(),
        "avg_fraud": df["avg_fraud"].tolist(),
        "max_anomaly": df["max_anomaly"].tolist(),
    }

@app.get("/charts/claimant_timeline")
def claimant_timeline(claimant_id: int, limit: int = 200):
    df = pd.read_csv(DATA_PROCESSED_DIR / "sim_claimant_timeseries.csv",
                     parse_dates=["timestamp"])
    df = df[df["claimant_id"] == claimant_id].sort_values("timestamp").tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
        "gnn_risk": df["gnn_risk"].tolist(),
    }

@app.get("/charts/gnn_clusters")
def gnn_clusters():
    df = pd.read_csv(DATA_PROCESSED_DIR / "sim_gnn_clusters.csv")
    return {
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "cluster": df["cluster"].tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "claim_id": df["claim_id"].tolist(),
    }

