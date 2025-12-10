# src/analytics/prepare_simulation_data.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import lightgbm as lgb

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR

def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Load training table (already joined with AE + TT + GNN)
    # -------------------------------------------------------------
    training_path = DATA_PROCESSED_DIR / "training_table.csv"
    df = pd.read_csv(training_path)

    # -------------------------------------------------------------
    # Load LightGBM model to compute fraud_score
    # -------------------------------------------------------------
    model_path = MODELS_DIR / "fraud_lgbm.txt"
    lgb_model = lgb.Booster(model_file=str(model_path))

    label_col = "is_fraud"
    exclude = ["claim_id", "claimant_id", "provider_id", label_col]
    feature_cols = [c for c in df.columns if c not in exclude]

    print("Computing LightGBM fraud scores...")
    df["fraud_score"] = lgb_model.predict(df[feature_cols])

    # AE anomaly score exists already in df["ae_score"]
    # We define anomaly_score = ae_score
    df["anomaly_score"] = df["ae_score"]

    # -------------------------------------------------------------
    # Create current timestamp-based time series
    # -------------------------------------------------------------
    print("Assigning timestamps...")
    base = pd.Timestamp.today().normalize()  # midnight today
    df = df.sort_values("claim_id").reset_index(drop=True)
    df["timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(len(df))]

    # Save claim-level time series
    ts_path = DATA_PROCESSED_DIR / "sim_claim_timeseries.csv"
    df_ts = df[[
        "claim_id", "claimant_id", "provider_id", "timestamp",
        "fraud_score", "anomaly_score"
    ]]
    df_ts.to_csv(ts_path, index=False)
    print(f"Wrote {ts_path}")

    # -------------------------------------------------------------
    # Provider Risk (Top-Level Aggregation)
    # -------------------------------------------------------------
    print("Computing provider risk table...")
    provider_risk = (
        df_ts.groupby("provider_id")
        .agg(
            avg_fraud=("fraud_score", "mean"),
            max_anomaly=("anomaly_score", "max"),
        )
        .reset_index()
    )

    # Combined risk = LightGBM probability + anomaly score
    provider_risk["combined_risk"] = (
        provider_risk["avg_fraud"] + provider_risk["max_anomaly"]
    )

    prov_path = DATA_PROCESSED_DIR / "sim_provider_risk.csv"
    provider_risk.to_csv(prov_path, index=False)
    print(f"Wrote {prov_path}")

    # -------------------------------------------------------------
    # Claimant timelines
    # -------------------------------------------------------------
    print("Writing claimant-level time series...")
    claimant_ts_path = DATA_PROCESSED_DIR / "sim_claimant_timeseries.csv"
    df_ts.to_csv(claimant_ts_path, index=False)
    print(f"Wrote {claimant_ts_path}")
    
    # -------------------------------------------------------------
    # GNN cluster visualization (DBSCAN anomaly + fraud-ring injection)
    # -------------------------------------------------------------
    print("Preparing GNN anomaly visualization with synthetic outliers...")
    
    # Extract GNN embedding columns
    gnn_cols = [c for c in df.columns if c.startswith("gnn_emb_")]
    X = df[gnn_cols].values.astype(float)
    
    # -------------------------------
    # STEP 1: Inject synthetic anomalies
    # -------------------------------
    num_nodes = X.shape[0]
    num_anomalies = max(5, num_nodes // 50)   # ≈ 2% anomalies
    rng = np.random.default_rng(42)
    anomaly_indices = rng.choice(num_nodes, size=num_anomalies, replace=False)
    
    # Inject large random noise into selected embeddings
    # Makes them stand far away in embedding space
    X[anomaly_indices] += rng.normal(loc=15.0, scale=4.0, size=X[anomaly_indices].shape)
    
    # Also optionally inject a small "fraud ring" micro-cluster
    num_ring = max(3, num_nodes // 100)       # ~1%
    ring_indices = rng.choice(num_nodes, size=num_ring, replace=False)
    ring_center = rng.normal(loc=8.0, scale=1.0, size=X.shape[1])
    X[ring_indices] = ring_center + rng.normal(loc=0.0, scale=0.5, size=X[ring_indices].shape)
    
    # -------------------------------
    # STEP 2: PCA to 2D
    # -------------------------------
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    
    # -------------------------------
    # STEP 3: DBSCAN anomaly detection
    # -------------------------------
    dbscan = DBSCAN(eps=2, min_samples=6)
    labels = dbscan.fit_predict(X2)
    
    # outlier flag
    outlier = (labels == -1).astype(int)
    
    # -------------------------------
    # STEP 4: Build visualization table
    # -------------------------------
    gnn_vis = pd.DataFrame({
        "x": X2[:, 0],
        "y": X2[:, 1],
        "cluster": labels,
        "outlier": outlier,
        "fraud_score": df["fraud_score"],
        "claim_id": df["claim_id"],
    })
    
    gnn_path = DATA_PROCESSED_DIR / "sim_gnn_clusters.csv"
    gnn_vis.to_csv(gnn_path, index=False)
    print(f"Wrote {gnn_path}")

    print("Simulation data preparation complete.")

if __name__ == "__main__":
    main()

