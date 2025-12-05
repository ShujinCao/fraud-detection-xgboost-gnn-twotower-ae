import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    N_CLAIMANTS,
    N_PROVIDERS,
    EMBED_DIM,
)
from src.twotower.model import TwoTowerModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ClaimsInteractionDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(DATA_RAW_DIR / "claims.csv")
        self.claim_ids = df["claim_id"].values
        self.claimant_ids = df["claimant_id"].values
        self.provider_ids = df["provider_id"].values
        self.labels = df["is_fraud"].values.astype("float32")

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        return (
            self.claim_ids[idx],
            int(self.claimant_ids[idx]),
            int(self.provider_ids[idx]),
            self.labels[idx],
        )

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ClaimsInteractionDataset()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(N_CLAIMANTS, N_PROVIDERS, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Training two-tower interaction model...")
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        for _, claimant_ids, provider_ids, labels in loader:
            claimant_ids = claimant_ids.to(device)
            provider_ids = provider_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(claimant_ids, provider_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "twotower.pt")

    # Extract per-claim embeddings
    model.eval()
    all_rows = []
    with torch.no_grad():
        for claim_id, claimant_id, provider_id, _ in dataset:
            c_id = torch.tensor([claimant_id], dtype=torch.long, device=device)
            p_id = torch.tensor([provider_id], dtype=torch.long, device=device)
            _, u_emb, p_emb = model(c_id, p_id)
            emb = torch.cat([u_emb, p_emb], dim=1).cpu().numpy().flatten()
            row = {"claim_id": int(claim_id)}
            for j, v in enumerate(emb):
                row[f"tt_emb_{j}"] = float(v)
            all_rows.append(row)

    df_emb = pd.DataFrame(all_rows)
    df_emb.to_csv(DATA_PROCESSED_DIR / "twotower_embeddings.csv", index=False)
    logger.info("Wrote two-tower embeddings to data/processed/twotower_embeddings.csv")

if __name__ == "__main__":
    main()

