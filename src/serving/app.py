# src/serving/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import lightgbm as lgb

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="FraudForge API")

# -------------------------------------------------------------
# Root route (fix for Render — ensures 200 OK at "/")
# -------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "FraudForge API is running"}

# -------------------------------------------------------------
# Lazy-loaded caches
# -------------------------------------------------------------
training_df = None
fraud_ts_df = None
claimant_ts_cache = {}
provider_risk_df = None
gnn_df = None
lgb_model = None

# -------------------------------------------------------------
# Lazy-load helpers
# -------------------------------------------------------------
def load_training_table():
    global training_df
    if training_df is None:
        logger.info("Loading training table...")
        training_df = pd.read_csv(DATA_PROCESSED_DIR / "training_table.csv")
    return training_df

def load_fraud_timeseries():
    global fraud_ts_df
    if fraud_ts_df is None:
        logger.info("Loading sim_claim_timeseries.csv...")
        fraud_ts_df = pd.read_csv(
            DATA_PROCESSED_DIR / "sim_claim_timeseries.csv",
            parse_dates=["timestamp"]
        )
    return fraud_ts_df

def load_claimant_timeseries(claimant_id: int):
    global claimant_ts_cache
    if claimant_id not in claimant_ts_cache:
        logger.info(f"Loading claimant timeline for {claimant_id}...")
        df = pd.read_csv(
            DATA_PROCESSED_DIR / "sim_claimant_timeseries.csv",
            parse_dates=["timestamp"]
        )
        df = df[df["claimant_id"] == claimant_id].sort_values("timestamp")
        claimant_ts_cache[claimant_id] = df
    return claimant_ts_cache[claimant_id]

def load_provider_risk():
    global provider_risk_df
    if provider_risk_df is None:
        logger.info("Loading sim_provider_risk.csv...")
        provider_risk_df = pd.read_csv(DATA_PROCESSED_DIR / "sim_provider_risk.csv")
    return provider_risk_df

def load_gnn_clusters():
    global gnn_df
    if gnn_df is None:
        logger.info("Loading sim_gnn_clusters.csv (DBSCAN results)...")
        gnn_df = pd.read_csv(DATA_PROCESSED_DIR / "sim_gnn_clusters.csv")
    return gnn_df

def get_lgb_model():
    global lgb_model
    if lgb_model is None:
        logger.info("Loading LightGBM model fraud_lgbm.txt...")
        lgb_model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_lgbm.txt"))
    return lgb_model

# -------------------------------------------------------------
# API schema
# -------------------------------------------------------------
class ClaimFeatures(BaseModel):
    claim_id: int
    claimant_id: int
    provider_id: int
    claim_amount: float
    procedure_code: int
    days_since_last_claim: int

# -------------------------------------------------------------
# Health check
# -------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------------
# Direct LGB scoring demo
# -------------------------------------------------------------
@app.post("/score_sync")
def score_sync(claim: ClaimFeatures):
    df_train = load_training_table()

    feature_cols = [
        c for c in df_train.columns 
        if c not in ["claim_id", "claimant_id", "provider_id", "is_fraud"]
    ]

    row = df_train[df_train["claim_id"] == claim.claim_id]

    if row.empty:
        feature_values = [0.0] * len(feature_cols)
    else:
        feature_values = row.iloc[0][feature_cols].values

    X = pd.DataFrame([feature_values], columns=feature_cols)
    model = get_lgb_model()
    prob = float(model.predict(X)[0])

    return {"fraud_risk_score": prob}

# -------------------------------------------------------------
# Streaming Dashboard Endpoints
# -------------------------------------------------------------
@app.get("/charts/fraud_trend")
def fraud_trend(limit: int = 500):
    df = load_fraud_timeseries().sort_values("timestamp").tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
        "claimant_id": df["claimant_id"].tolist(),
    }

@app.get("/charts/anomalies")
def anomalies(limit: int = 500, threshold: float = 0.95):
    df = load_fraud_timeseries().sort_values("timestamp").tail(limit)
    high = df[df["anomaly_score"] >= df["anomaly_score"].quantile(threshold)]
    return {
        "timestamp": high["timestamp"].astype(str).tolist(),
        "anomaly_score": high["anomaly_score"].tolist(),
        "claim_id": high["claim_id"].tolist(),
    }

@app.get("/charts/provider_risk")
def provider_risk():
    df = load_provider_risk()
    df_top = df.sort_values("combined_risk", ascending=False).head(10)

    return {
        "provider_id": df_top["provider_id"].tolist(),
        "avg_fraud": df_top["avg_fraud"].tolist(),
        "max_anomaly": df_top["max_anomaly"].tolist(),
        "combined_risk": df_top["combined_risk"].tolist(),
    }

@app.get("/charts/claimant_timeline")
def claimant_timeline(claimant_id: int, limit: int = 200):
    df = load_claimant_timeseries(claimant_id).tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
    }

@app.get("/charts/gnn_clusters")
def gnn_clusters():
    df = load_gnn_clusters()
    return {
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "cluster": df["cluster"].tolist(),
        "outlier": df["outlier"].tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "claim_id": df["claim_id"].tolist(),
    }

