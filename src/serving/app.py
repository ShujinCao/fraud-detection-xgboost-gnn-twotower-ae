from fastapi import FastAPI
import pandas as pd
import xgboost as xgb
from pathlib import Path

from src.serving.schema import ClaimFeatures
from src.config import MODELS_DIR, DATA_PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Fraud Risk Scoring API")

# Load training schema to keep feature ordering consistent
training_table_path = DATA_PROCESSED_DIR / "training_table.csv"
df_train = pd.read_csv(training_table_path)
LABEL_COL = "is_fraud"
EXCLUDE_COLS = ["claim_id", "claimant_id", "provider_id", LABEL_COL]
FEATURE_COLS = [c for c in df_train.columns if c not in EXCLUDE_COLS]

model_path = MODELS_DIR / "xgb.json"
xgb_model = xgb.Booster()
xgb_model.load_model(model_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(claim: ClaimFeatures):
    # Build single-row feature frame
    row = {
        "claim_id": claim.claim_id,
        "claimant_id": claim.claimant_id,
        "provider_id": claim.provider_id,
        "claim_amount": claim.claim_amount,
        "procedure_code": claim.procedure_code,
        "days_since_last_claim": claim.days_since_last_claim,
    }

    # We don't recompute embeddings here for demo; instead,
    # look up the precomputed row in training_table (idempotent showcase).
    df_row = df_train[df_train["claim_id"] == claim.claim_id]

    if df_row.empty:
        # Fallback: just fill with zeros if claim_id unknown
        feature_values = [0.0] * len(FEATURE_COLS)
    else:
        feature_values = df_row.iloc[0][FEATURE_COLS].values

    dmatrix = xgb.DMatrix([feature_values], feature_names=FEATURE_COLS)
    prob = float(xgb_model.predict(dmatrix)[0])
    return {"fraud_risk_score": prob}

