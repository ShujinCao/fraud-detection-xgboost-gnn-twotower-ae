import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.pipeline.join_embeddings import main as join_main
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure joined table is up-to-date
    join_main()

    df = pd.read_csv(DATA_PROCESSED_DIR / "training_table.csv")
    label_col = "is_fraud"

    feature_cols = [c for c in df.columns if c not in ["claim_id", "claimant_id", "provider_id", label_col]]
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    logger.info("Training XGBoost model...")
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False
    )

    y_val_pred = bst.predict(dval)
    auc = roc_auc_score(y_val, y_val_pred)
    logger.info(f"Validation AUC: {auc:.4f}")

    bst.save_model(MODELS_DIR / "xgb.json")
    logger.info("Saved XGBoost model to models/xgb.json")

if __name__ == "__main__":
    main()

