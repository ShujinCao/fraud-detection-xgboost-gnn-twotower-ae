import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np

from src.autoencoder.dataset import ClaimsAEDataset, AE_FEATURE_COLS
from src.autoencoder.model import Autoencoder
from src.config import MODELS_DIR, DATA_PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ClaimsAEDataset()
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=len(AE_FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logger.info("Training autoencoder (MSE reconstruction loss)...")
    model.train()
    for epoch in range(30):
        epoch_loss = 0.0
        for _, batch_x in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, z = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(dataset):.6f}")

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    logger.info("Saved AE model")

    # --- Export per-claim AE features: reconstruction error + latent embedding ---
    model.eval()
    claim_ids_all = []
    mse_all = []
    latent_all = []

    with torch.no_grad():
        for claim_ids, batch_x in loader:
            batch_x = batch_x.to(device)
            x_hat, z = model(batch_x)
            mse = ((x_hat - batch_x) ** 2).mean(dim=1)

            claim_ids_all.extend(claim_ids.numpy().tolist())
            mse_all.extend(mse.cpu().numpy().tolist())
            latent_all.append(z.cpu().numpy())

    latent_all = np.concatenate(latent_all, axis=0)

    rows = []
    for cid, mse_val, z_vec in zip(claim_ids_all, mse_all, latent_all):
        row = {"claim_id": int(cid), "ae_score": float(mse_val)}
        for j, v in enumerate(z_vec):
            row[f"ae_latent_{j}"] = float(v)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROCESSED_DIR / "ae_features.csv", index=False)
    logger.info("Wrote AE features (score + latent) to data/processed/ae_features.csv")

if __name__ == "__main__":
    main()

