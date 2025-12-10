import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv

from pathlib import Path

# Adjust this to match your repo layout
DATA_PROCESSED_DIR = Path("data/processed")


# -----------------------------
# 1. Create synthetic medical graph
# -----------------------------
def build_medical_graph(rng=np.random.default_rng(42)):
    num_claimants = 500
    num_providers = 80
    num_devices = 150
    num_ips = 120

    CLAIMANT_OFFSET = 0
    PROVIDER_OFFSET = CLAIMANT_OFFSET + num_claimants
    DEVICE_OFFSET = PROVIDER_OFFSET + num_providers
    IP_OFFSET = DEVICE_OFFSET + num_devices
    TOTAL_NODES = IP_OFFSET + num_ips

    G = nx.Graph()

    # Node types
    for i in range(num_claimants):
        G.add_node(CLAIMANT_OFFSET + i, ntype="claimant")
    for i in range(num_providers):
        G.add_node(PROVIDER_OFFSET + i, ntype="provider")
    for i in range(num_devices):
        G.add_node(DEVICE_OFFSET + i, ntype="device")
    for i in range(num_ips):
        G.add_node(IP_OFFSET + i, ntype="ip")

    # Provider specialties & regions
    specialties = ["radiology", "orthopedics", "dermatology", "general"]
    regions = ["north", "south", "east", "west"]
    spec_to_idx = {s: i for i, s in enumerate(specialties)}
    reg_to_idx = {r: i for i, r in enumerate(regions)}

    for p in range(num_providers):
        spec = specialties[rng.integers(len(specialties))]
        reg = regions[rng.integers(len(regions))]
        node_id = PROVIDER_OFFSET + p
        G.nodes[node_id]["specialty"] = spec
        G.nodes[node_id]["region"] = reg

    # helper functions
    def random_claimant():
        return CLAIMANT_OFFSET + rng.integers(num_claimants)
    def random_provider():
        return PROVIDER_OFFSET + rng.integers(num_providers)
    def random_device():
        return DEVICE_OFFSET + rng.integers(num_devices)
    def random_ip():
        return IP_OFFSET + rng.integers(num_ips)

    # -----------------------------
    # Normal medical claim behavior
    # -----------------------------
    num_normal_claims = 1500
    for _ in range(num_normal_claims):
        c = random_claimant()
        p = random_provider()
        d = random_device()
        ip = random_ip()
        G.add_edge(c, p, etype="normal_claim")
        G.add_edge(c, d, etype="device")
        G.add_edge(c, ip, etype="ip")

    # -----------------------------
    # Fraud ring (tight micro-cluster)
    # -----------------------------
    fraud_ring_claimants = rng.choice(
        np.arange(CLAIMANT_OFFSET, CLAIMANT_OFFSET + num_claimants),
        size=20,
        replace=False
    )
    fraud_ring_provider = PROVIDER_OFFSET + rng.integers(num_providers)
    fraud_ring_devices = rng.choice(
        np.arange(DEVICE_OFFSET, DEVICE_OFFSET + num_devices),
        size=3,
        replace=False
    )
    fraud_ring_ips = rng.choice(
        np.arange(IP_OFFSET, IP_OFFSET + num_ips),
        size=2,
        replace=False
    )

    for c in fraud_ring_claimants:
        G.add_edge(c, fraud_ring_provider, etype="fraud_claim")
        for d in fraud_ring_devices:
            G.add_edge(c, d, etype="fraud_device")
        for ip in fraud_ring_ips:
            G.add_edge(c, ip, etype="fraud_ip")

    # -----------------------------
    # Synthetic identity outliers
    # -----------------------------
    num_outliers = 15
    outlier_claimants = rng.choice(
        np.arange(CLAIMANT_OFFSET, CLAIMANT_OFFSET + num_claimants),
        size=num_outliers,
        replace=False
    )

    for c in outlier_claimants:
        # Very sparse, weird connectivity
        p = random_provider()
        ip = random_ip()
        G.add_edge(c, p, etype="odd_claim")
        G.add_edge(c, ip, etype="odd_ip")

    # -----------------------------
    # Build node features
    # -----------------------------
    ntype_to_idx = {"claimant": 0, "provider": 1, "device": 2, "ip": 3}
    X = np.zeros((TOTAL_NODES, 4 + len(specialties) + len(regions) + 1), dtype=float)
    # last extra dim for degree

    for n in range(TOTAL_NODES):
        ntype = G.nodes[n].get("ntype", "claimant")
        X[n, ntype_to_idx[ntype]] = 1.0
        # degree feature
        X[n, -1] = G.degree[n]
        if ntype == "provider":
            spec = G.nodes[n].get("specialty", None)
            reg = G.nodes[n].get("region", None)
            if spec is not None:
                X[n, 4 + spec_to_idx[spec]] = 1.0
            if reg is not None:
                X[n, 4 + len(specialties) + reg_to_idx[reg]] = 1.0

    # -----------------------------
    # Fraud labels for training (node classification)
    # -----------------------------
    y = np.zeros(TOTAL_NODES, dtype=int)  # 0 = non-fraud, 1 = fraud
    for c in fraud_ring_claimants:
        y[c] = 1
    y[fraud_ring_provider] = 1
    for c in outlier_claimants:
        y[c] = 1

    # Ground-truth semantic segments for visualization
    # (these are for explanation, not for training DBSCAN)
    segment_label = np.array(["normal"] * TOTAL_NODES, dtype=object)
    for p in range(num_providers):
        node_id = PROVIDER_OFFSET + p
        spec = G.nodes[node_id].get("specialty", "general")
        segment_label[node_id] = spec  # e.g. "radiology"
    segment_label[fraud_ring_provider] = "fraud_ring_provider"
    for c in fraud_ring_claimants:
        segment_label[c] = "fraud_ring_claimant"
    for c in outlier_claimants:
        segment_label[c] = "synthetic_identity"

    meta = {
        "CLAIMANT_OFFSET": CLAIMANT_OFFSET,
        "PROVIDER_OFFSET": PROVIDER_OFFSET,
        "DEVICE_OFFSET": DEVICE_OFFSET,
        "IP_OFFSET": IP_OFFSET,
        "TOTAL_NODES": TOTAL_NODES,
    }

    return G, X, y, segment_label, meta


# -----------------------------
# 2. Define GraphSAGE model
# -----------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


def train_gnn(G, X, y, epochs=80, hidden_dim=64, lr=1e-3, weight_decay=5e-4):
    data = from_networkx(G)
    data.x = torch.tensor(X, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.long)

    TOTAL_NODES = data.x.size(0)
    idx = np.arange(TOTAL_NODES)
    np.random.shuffle(idx)
    train_size = int(0.6 * TOTAL_NODES)
    val_size = int(0.2 * TOTAL_NODES)
    train_idx = torch.tensor(idx[:train_size], dtype=torch.long)
    val_idx = torch.tensor(idx[train_size:train_size + val_size], dtype=torch.long)
    test_idx = torch.tensor(idx[train_size + val_size:], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(data.x.size(1), hidden_dim, 2).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_split(split_idx):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            correct = (preds[split_idx] == data.y[split_idx]).sum().item()
            return correct / split_idx.size(0)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch()
        if epoch % 10 == 0:
            train_acc = eval_split(train_idx)
            val_acc = eval_split(val_idx)
            print(f"Epoch {epoch:03d} | Loss={loss:.4f} | Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index).cpu().numpy()
        logits = model(data.x, data.edge_index).cpu().numpy()
        fraud_prob = torch.softmax(
            model(data.x.to(device), data.edge_index), dim=1
        )[:, 1].cpu().numpy()

    return emb, fraud_prob


def main():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building synthetic structured medical claims graph...")
    G, X, y, segment_label, meta = build_medical_graph()
    print("Training GraphSAGE GNN...")
    emb, fraud_prob = train_gnn(G, X, y)

    print("Running PCA on embeddings...")
    pca = PCA(n_components=2)
    emb2d = pca.fit_transform(emb)

    print("Running DBSCAN for anomaly detection...")
    dbscan = DBSCAN(eps=3.0, min_samples=5)
    labels = dbscan.fit_predict(emb2d)
    outlier = (labels == -1).astype(int)

    # We'll focus on claimant nodes (like "claims") for the viz
    CLAIMANT_OFFSET = meta["CLAIMANT_OFFSET"]
    PROVIDER_OFFSET = meta["PROVIDER_OFFSET"]
    num_claimants = PROVIDER_OFFSET - CLAIMANT_OFFSET

    claimant_idx = np.arange(CLAIMANT_OFFSET, PROVIDER_OFFSET)

    df_vis = pd.DataFrame({
        "node_id": claimant_idx,
        "x": emb2d[claimant_idx, 0],
        "y": emb2d[claimant_idx, 1],
        "cluster": labels[claimant_idx],
        "outlier": outlier[claimant_idx],
        "fraud_score": fraud_prob[claimant_idx],
        "fraud_label": y[claimant_idx],
        "segment_label": segment_label[claimant_idx],
    })

    # For compatibility with your existing pipeline:
    # treat "node_id" as "claim_id" for visual purposes
    df_vis["claim_id"] = df_vis["node_id"]

    out_path = DATA_PROCESSED_DIR / "sim_gnn_clusters.csv"
    df_vis[["x", "y", "cluster", "outlier", "fraud_score", "claim_id", "segment_label"]].to_csv(
        out_path, index=False
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

