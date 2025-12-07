import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR
import lightgbm as lgb

def main():
    # 1) Load training table
    training_path = DATA_PROCESSED_DIR / "training_table.csv"
    df = pd.read_csv(training_path)

    # 2) Load LightGBM model to get fraud scores (probabilities)
    model_path = MODELS_DIR / "fraud_lgbm.txt"
    lgb_model = lgb.Booster(model_file=str(model_path))

    label_col = "is_fraud"
    exclude_cols = ["claim_id", "claimant_id", "provider_id", label_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    probs = lgb_model.predict(df[feature_cols])
    df["fraud_score"] = probs

    # 3) AE anomaly score: you already have ae_score in AE features
    # we assume it is merged into training_table.csv as ae_score
    if "ae_score" not in df.columns:
        raise ValueError("ae_score column not found in training_table.csv")

    df["anomaly_score"] = df["ae_score"]

    # 4) Simple GNN risk proxy: norm of GNN embedding, or first component
    gnn_cols = [c for c in df.columns if c.startswith("gnn_emb_")]
    if gnn_cols:
        df["gnn_risk"] = np.linalg.norm(df[gnn_cols].values, axis=1)
    else:
        df["gnn_risk"] = 0.0

    # 5) Simulated timestamps (e.g., 1-minute steps)
    df = df.sort_values("claim_id")
    base = pd.Timestamp("2024-01-01 00:00:00")
    df["timestamp"] = [base + pd.Timedelta(minutes=i) for i in range(len(df))]

    # 6) Save "claim-level time series"
    ts_path = DATA_PROCESSED_DIR / "sim_claim_timeseries.csv"
    df_out = df[[
        "claim_id", "claimant_id", "provider_id", "timestamp",
        "fraud_score", "anomaly_score", "gnn_risk"
    ]]
    df_out.to_csv(ts_path, index=False)

    # 7) Provider-level risk (rolling mean)
    provider_risk = (
        df_out
        .groupby(["provider_id"])
        .agg(avg_fraud=("fraud_score", "mean"),
             max_anomaly=("anomaly_score", "max"))
        .reset_index()
    )
    provider_risk.to_csv(DATA_PROCESSED_DIR / "sim_provider_risk.csv", index=False)

    # 8) Claimant timelines (e.g., last N points per claimant)
    claimant_ts = df_out.copy()
    claimant_ts.to_csv(DATA_PROCESSED_DIR / "sim_claimant_timeseries.csv", index=False)

    # 9) GNN clusters (for visualization – static)
    if gnn_cols:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        X = df[gnn_cols].values
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)

        km = KMeans(n_clusters=4, random_state=42)
        clusters = km.fit_predict(X)

        gnn_vis = pd.DataFrame({
            "x": X2[:, 0],
            "y": X2[:, 1],
            "cluster": clusters,
            "fraud_score": df["fraud_score"],
            "claim_id": df["claim_id"],
        })
        gnn_vis.to_csv(DATA_PROCESSED_DIR / "sim_gnn_clusters.csv", index=False)

if __name__ == "__main__":
    main()

