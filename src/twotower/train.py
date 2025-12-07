import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import numpy as np

logger = get_logger(__name__)

class PositivePairsDataset(Dataset):
    """
    Each row is a positive (claimant, provider) pair from claims.
    """
    def __init__(self):
        df = pd.read_csv(DATA_RAW_DIR / "claims.csv")
        self.claim_ids = df["claim_id"].values
        self.claimant_ids = df["claimant_id"].values
        self.provider_ids = df["provider_id"].values

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        return (
            int(self.claim_ids[idx]),
            int(self.claimant_ids[idx]),
            int(self.provider_ids[idx]),
        )

def contrastive_loss(u, i_pos, i_neg):
    """
    u: (B, D) claimant embeddings
    i_pos: (B, D) positive provider embeddings
    i_neg: (B, K, D) negative provider embeddings
    """
    # positive logits: dot(u, i_pos)
    pos_logits = (u * i_pos).sum(dim=1)          # (B,)
    pos_logprob = F.logsigmoid(pos_logits)

    # negatives: dot(u, i_neg_k) for each k
    # u: (B, D) -> (B, 1, D), i_neg: (B, K, D)
    u_expanded = u.unsqueeze(1)                  # (B, 1, D)
    neg_logits = (u_expanded * i_neg).sum(dim=2) # (B, K)
    neg_logprob = F.logsigmoid(-neg_logits).sum(dim=1)  # (B,)

    loss = -(pos_logprob + neg_logprob).mean()
    return loss

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = PositivePairsDataset()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(N_CLAIMANTS, N_PROVIDERS, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_negatives = 5
    all_provider_ids = np.arange(1, N_PROVIDERS + 1)

    logger.info("Training two-tower with unsupervised contrastive loss...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for _, claimant_ids, provider_ids in loader:
            claimant_ids = claimant_ids.to(device)
            provider_ids_pos = provider_ids.to(device)

            # sample negatives for each batch row
            neg_ids = np.random.choice(all_provider_ids, size=(len(claimant_ids), num_negatives), replace=True)
            neg_ids = torch.tensor(neg_ids, dtype=torch.long, device=device)

            u = model.claimant_emb(claimant_ids)
            i_pos = model.provider_emb(provider_ids_pos)
            i_neg = model.provider_emb(neg_ids)  # (B, K, D)

            # normalize for stability
            u = F.normalize(u, p=2, dim=1)
            i_pos = F.normalize(i_pos, p=2, dim=1)
            i_neg = F.normalize(i_neg, p=2, dim=2)

            loss = contrastive_loss(u, i_pos, i_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * claimant_ids.size(0)

        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "twotower.pt")
    logger.info("Saved two-tower model")

    # --- Export per-claim embedding: concat claimant + provider emb ---
    df = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    model.eval()
    rows = []
    with torch.no_grad():
        for _, row in df.iterrows():
            cid = int(row["claim_id"])
            cl_id = int(row["claimant_id"])
            pr_id = int(row["provider_id"])

            cl_t = torch.tensor([cl_id], dtype=torch.long, device=device)
            pr_t = torch.tensor([pr_id], dtype=torch.long, device=device)
            u = F.normalize(model.claimant_emb(cl_t), p=2, dim=1)
            i = F.normalize(model.provider_emb(pr_t), p=2, dim=1)

            emb = torch.cat([u, i], dim=1).cpu().numpy().flatten()
            out_row = {"claim_id": cid}
            for j, v in enumerate(emb):
                out_row[f"tt_emb_{j}"] = float(v)
            rows.append(out_row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROCESSED_DIR / "twotower_embeddings.csv", index=False)
    logger.info("Wrote two-tower embeddings to data/processed/twotower_embeddings.csv")

if __name__ == "__main__":
    main()

