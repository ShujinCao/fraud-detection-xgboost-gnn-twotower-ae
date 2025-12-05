import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from src.config import DATA_RAW_DIR, RANDOM_SEED

def build_bipartite_adjacency():
    """
    Build a simple bipartite graph between claimants and providers.
    We index:
        0 .. (n_claimants-1)        -> claimant nodes
        n_claimants .. end          -> provider nodes
    """
    claims = pd.read_csv(DATA_RAW_DIR / "claims.csv")
    n_claimants = claims["claimant_id"].max()
    n_providers = claims["provider_id"].max()

    row = []
    col = []

    for _, row_data in claims.iterrows():
        c = int(row_data["claimant_id"]) - 1
        p = n_providers + int(row_data["provider_id"]) - 1  # offset providers
        row.extend([c, p])
        col.extend([p, c])

    n_nodes = n_claimants + n_providers
    data = np.ones(len(row))

    adj = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    return adj, n_claimants, n_providers

