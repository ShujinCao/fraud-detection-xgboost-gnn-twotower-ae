import pandas as pd
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    claims = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    ae = pd.read_csv(DATA_PROCESSED_DIR / "ae_features.csv")          # updated
    tt = pd.read_csv(DATA_PROCESSED_DIR / "twotower_embeddings.csv")
    gnn = pd.read_csv(DATA_PROCESSED_DIR / "gnn_embeddings.csv")

    df = claims.merge(ae, on="claim_id", how="left") \
               .merge(tt, on="claim_id", how="left") \
               .merge(gnn, on="claim_id", how="left")

    df.to_csv(DATA_PROCESSED_DIR / "training_table.csv", index=False)
    logger.info("Wrote joined training table to data/processed/training_table.csv")

if __name__ == "__main__":
    main()

