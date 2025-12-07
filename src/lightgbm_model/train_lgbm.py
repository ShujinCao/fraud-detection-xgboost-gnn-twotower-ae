import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.pipeline.join_embeddings import main as join_main
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ensure the joined table exists
    join_main()

    df = pd.read_csv(DATA_PROCESSED_DIR / "training_table.csv")
    label_col = "is_fraud"

    exclude_cols = ["claim_id", "claimant_id", "provider_id", label_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    logger.info("Training LightGBM classifier on AE + TT + GNN + raw features...")
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),],
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_val_pred)
    logger.info(f"Validation AUC: {auc:.4f}")

    model.save_model(str(MODELS_DIR / "fraud_lgbm.txt"))
    logger.info("Saved LightGBM model to models/fraud_lgbm.txt")

if __name__ == "__main__":
    main()

