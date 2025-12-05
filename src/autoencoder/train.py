import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd

from src.autoencoder.dataset import ClaimsAEDataset, AE_FEATURE_COLS
from src.autoencoder.model import Autoencoder
from src.config import MODELS_DIR, DATA_PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ClaimsAEDataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=len(AE_FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logger.info("Training autoencoder...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for _, batch_x in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        logger.info(f"Epoch {epoch+1}: loss={epoch_loss/len(dataset):.6f}")

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    logger.info("Saved AE model")

    # Compute reconstruction errors for each claim
    model.eval()
    claim_ids = []
    scores = []
    with torch.no_grad():
        for ids, batch_x in loader:
            batch_x = batch_x.to(device)
            x_hat, _ = model(batch_x)
            mse = ((x_hat - batch_x) ** 2).mean(dim=1)
            claim_ids.extend(ids.numpy().tolist())
            scores.extend(mse.cpu().numpy().tolist())

    df_scores = pd.DataFrame({
        "claim_id": claim_ids,
        "ae_score": scores,
    })
    df_scores.to_csv(DATA_PROCESSED_DIR / "ae_scores.csv", index=False)
    logger.info("Wrote AE scores to data/processed/ae_scores.csv")

if __name__ == "__main__":
    main()

