import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DATA_RAW_DIR

AE_FEATURE_COLS = [
    "claim_amount",
    "procedure_code",
    "days_since_last_claim",
    "claimant_base_risk",
    "provider_base_risk",
]

class ClaimsAEDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(DATA_RAW_DIR / "claims.csv")
        self.claim_ids = df["claim_id"].values
        x = df[AE_FEATURE_COLS].values.astype("float32")
        self.x = torch.from_numpy(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.claim_ids[idx], self.x[idx]

