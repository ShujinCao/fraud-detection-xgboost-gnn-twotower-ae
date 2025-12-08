# src/serving/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import lightgbm as lgb

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="FraudForge Simulation API")

# ------------------------------------------------------------
# Root route (IMPORTANT for Render "Application loading" fix)
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "FraudForge API is running"}

# ------------------------------------------------------------
# Lazy-loaded global caches
# ------------------------------------------------------------
training_df = None
fraud_ts_df = None
claimant_ts_df_cache = {}
provider_risk_df = None
gnn_cluster_df = None
lgb_model = None

# ------------------------------------------------------------
# Utility lazy-loaders
# ------------------------------------------------------------
def load_training_table():
    global training_df
    if training_df is None:
        logger.info("Loading training_table.csv...")
        training_df = pd.read_csv(DATA_PROCESSED_DIR / "training_table.csv")
    return training_df

def load_fraud_timeseries():
    global fraud_ts_df
    if fraud_ts_df is None:
        logger.info("Loading sim_claim_timeseries.csv...")
        fraud_ts_df = pd.read_csv(
            DATA_PROCESSED_DIR / "sim_claim_timeseries.csv",
            parse_dates=["timestamp"],
        )
    return fraud_ts_df

def load_claimant_timeseries(claimant_id: int):
    global claimant_ts_df_cache
    if claimant_id not in claimant_ts_df_cache:
        logger.info(f"Loading claimant timeline for {claimant_id}...")
        df = pd.read_csv(
            DATA_PROCESSED_DIR / "sim_claimant_timeseries.csv",
            parse_dates=["timestamp"],
        )
        df = df[df["claimant_id"] == claimant_id]
        claimant_ts_df_cache[claimant_id] = df.sort_values("timestamp")
    return claimant_ts_df_cache[claimant_id]

def load_provider_risk():
    global provider_risk_df
    if provider_risk_df is None:
        logger.info("Loading sim_provider_risk.csv...")
        provider_risk_df = pd.read_csv(DATA_PROCESSED_DIR / "sim_provider_risk.csv")
    return provider_risk_df

def load_gnn_clusters():
    global gnn_cluster_df
    if gnn_cluster_df is None:
        logger.info("Loading sim_gnn_clusters.csv...")
        gnn_cluster_df = pd.read_csv(DATA_PROCESSED_DIR / "sim_gnn_clusters.csv")
    return gnn_cluster_df

def get_lgb_model():
    global lgb_model
    if lgb_model is None:
        logger.info("Loading LightGBM model fraud_lgbm.txt...")
        model_path = MODELS_DIR / "fraud_lgbm.txt"
        lgb_model = lgb.Booster(model_file=str(model_path))
    return lgb_model

# ------------------------------------------------------------
# API Schemas
# ------------------------------------------------------------
class ClaimFeatures(BaseModel):
    claim_id: int
    claimant_id: int
    provider_id: int
    claim_amount: float
    procedure_code: int
    days_since_last_claim: int

# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------
# Synchronous scoring endpoint
# (NOT used by HF dashboard, but useful for demo)
# ------------------------------------------------------------
@app.post("/score_sync")
def score_sync(claim: ClaimFeatures):
    df_train = load_training_table()
    feature_cols = [c for c in df_train.columns if c not in 
                    ["claim_id", "claimant_id", "provider_id", "is_fraud"]]

    row = df_train[df_train["claim_id"] == claim.claim_id]

    if row.empty:
        feature_values = [0.0] * len(feature_cols)
    else:
        feature_values = row.iloc[0][feature_cols].values

    X = pd.DataFrame([feature_values], columns=feature_cols)
    model = get_lgb_model()
    prob = float(model.predict(X)[0])

    return {"fraud_risk_score": prob}

# ------------------------------------------------------------
# Simulation Endpoints (HF Dashboard)
# ------------------------------------------------------------

@app.get("/charts/fraud_trend")
def fraud_trend(limit: int = 500):
    df = load_fraud_timeseries().sort_values("timestamp").tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
        "gnn_risk": df["gnn_risk"].tolist(),
        "claimant_id": df["claimant_id"].tolist(),  # useful for HF sample selection
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
    df = load_provider_risk().sort_values("avg_fraud", ascending=False).head(20)
    return {
        "provider_id": df["provider_id"].tolist(),
        "avg_fraud": df["avg_fraud"].tolist(),
        "max_anomaly": df["max_anomaly"].tolist(),
    }

@app.get("/charts/claimant_timeline")
def claimant_timeline(claimant_id: int, limit: int = 200):
    df = load_claimant_timeseries(claimant_id).tail(limit)
    return {
        "timestamp": df["timestamp"].astype(str).tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "anomaly_score": df["anomaly_score"].tolist(),
        "gnn_risk": df["gnn_risk"].tolist(),
    }

@app.get("/charts/gnn_clusters")
def gnn_clusters():
    df = load_gnn_clusters()
    return {
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "cluster": df["cluster"].tolist(),
        "fraud_score": df["fraud_score"].tolist(),
        "claim_id": df["claim_id"].tolist(),
    }

