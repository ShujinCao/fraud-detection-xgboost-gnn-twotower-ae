import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from src.gnn.model import GraphSAGE
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, EMBED_DIM, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_adj_lists():
    """
    Build bipartite adjacency between claimants and providers.
    Node indexing:
        0 .. (n_claimants-1) -> claimants
        n_claimants .. n_claimants+n_providers-1 -> providers
    """
    claims = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    n_claimants = claims["claimant_id"].max()
    n_providers = claims["provider_id"].max()

    num_nodes = n_claimants + n_providers
    adj_lists = [[] for _ in range(num_nodes)]

    for _, row in claims.iterrows():
        c = int(row["claimant_id"]) - 1
        p = n_claimants + int(row["provider_id"]) - 1
        adj_lists[c].append(p)
        adj_lists[p].append(c)

    return adj_lists, n_claimants, n_providers

def link_prediction_loss(z, edges_pos, num_nodes, num_neg=5):
    """
    Unsupervised contrastive loss:
      log σ(z_u · z_v) + Σ log σ(-z_u · z_v_neg)
    """
    device = z.device
    u_idx = torch.tensor([e[0] for e in edges_pos], dtype=torch.long, device=device)
    v_idx = torch.tensor([e[1] for e in edges_pos], dtype=torch.long, device=device)

    z_u = z[u_idx]  # (B, D)
    z_v = z[v_idx]  # (B, D)
    pos_logits = (z_u * z_v).sum(dim=1)
    pos_logprob = F.logsigmoid(pos_logits)

    # negatives: random mismatched nodes
    neg_v = torch.randint(0, num_nodes, (len(edges_pos), num_neg), device=device)
    z_v_neg = z[neg_v]  # (B, K, D)
    z_u_exp = z_u.unsqueeze(1)
    neg_logits = (z_u_exp * z_v_neg).sum(dim=2)
    neg_logprob = F.logsigmoid(-neg_logits).sum(dim=1)

    loss = -(pos_logprob + neg_logprob).mean()
    return loss

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    claims = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    adj_lists, n_claimants, n_providers = build_adj_lists()
    num_nodes = n_claimants + n_providers

    torch.manual_seed(RANDOM_SEED)
    x_init = torch.randn(num_nodes, EMBED_DIM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(in_dim=EMBED_DIM, hidden_dim=EMBED_DIM, out_dim=EMBED_DIM, num_layers=2).to(device)
    x_init = x_init.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Build positive edge list
    edges_pos = []
    for _, row in claims.iterrows():
        c = int(row["claimant_id"]) - 1
        p = n_claimants + int(row["provider_id"]) - 1
        edges_pos.append((c, p))

    logger.info("Training GraphSAGE with unsupervised link-prediction loss...")
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        z = model(x_init, adj_lists)  # (N, D)
        z = F.normalize(z, p=2, dim=1)
        loss = link_prediction_loss(z, edges_pos, num_nodes, num_neg=5)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}: loss={loss.item():.6f}")

    torch.save(model.state_dict(), MODELS_DIR / "gnn_graphsage.pt")
    logger.info("Saved GraphSAGE model")

    # --- Export per-claim GNN embedding: average claimant+provider node emb ---
    model.eval()
    with torch.no_grad():
        z_final = model(x_init, adj_lists)
        z_final = F.normalize(z_final, p=2, dim=1).cpu().numpy()

    rows = []
    for _, row in claims.iterrows():
        cid = int(row["claim_id"])
        c_idx = int(row["claimant_id"]) - 1
        p_idx = n_claimants + int(row["provider_id"]) - 1
        emb = (z_final[c_idx] + z_final[p_idx]) / 2.0
        out_row = {"claim_id": cid}
        for j, v in enumerate(emb):
            out_row[f"gnn_emb_{j}"] = float(v)
        rows.append(out_row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(DATA_PROCESSED_DIR / "gnn_embeddings.csv", index=False)
    logger.info("Wrote GraphSAGE embeddings to data/processed/gnn_embeddings.csv")

if __name__ == "__main__":
    main()

