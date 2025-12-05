import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from src.gnn.build_graph import build_bipartite_adjacency
from src.gnn.model import SimpleGCN
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, EMBED_DIM, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

def normalize_adj(adj: coo_matrix):
    adj = adj + coo_matrix(np.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = coo_matrix(
        (d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
        shape=adj.shape,
    )
    return D_inv_sqrt @ adj @ D_inv_sqrt

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    claims = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    adj, n_claimants, n_providers = build_bipartite_adjacency()
    logger.info(f"Graph has {adj.shape[0]} nodes")

    adj_norm = normalize_adj(adj).tocoo()
    indices = torch.tensor([adj_norm.row, adj_norm.col], dtype=torch.long)
    values = torch.tensor(adj_norm.data, dtype=torch.float32)
    adj_t = torch.sparse_coo_tensor(indices, values, size=adj_norm.shape)

    torch.manual_seed(RANDOM_SEED)
    x_init = torch.randn(adj.shape[0], EMBED_DIM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGCN(in_dim=EMBED_DIM, hidden_dim=EMBED_DIM, out_dim=EMBED_DIM).to(device)
    x_init = x_init.to(device)
    adj_t = adj_t.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Training simple GCN (unsupervised - smoothing embeddings)...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        z = model(x_init, adj_t)
        # simple regularization: encourage small magnitude embeddings
        loss = (z ** 2).mean()
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}: loss={loss.item():.6f}")

    torch.save(model.state_dict(), MODELS_DIR / "gnn.pt")

    model.eval()
    with torch.no_grad():
        z_final = model(x_init, adj_t).cpu().numpy()

    # Map claim_ids -> combined node embedding (average of claimant & provider node)
    rows = []
    for _, row in claims.iterrows():
        c_id = int(row["claimant_id"]) - 1
        p_id = n_providers + int(row["provider_id"]) - 1
        emb = (z_final[c_id] + z_final[p_id]) / 2.0
        row_dict = {"claim_id": int(row["claim_id"])}
        for j, v in enumerate(emb):
            row_dict[f"gnn_emb_{j}"] = float(v)
        rows.append(row_dict)

    df_emb = pd.DataFrame(rows)
    df_emb.to_csv(DATA_PROCESSED_DIR / "gnn_embeddings.csv", index=False)
    logger.info("Wrote GNN embeddings to data/processed/gnn_embeddings.csv")

if __name__ == "__main__":
    main()

