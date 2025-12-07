import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, n_claimants, n_providers, embed_dim):
        super().__init__()
        self.claimant_emb = nn.Embedding(n_claimants + 1, embed_dim)
        self.provider_emb = nn.Embedding(n_providers + 1, embed_dim)

    def forward(self, claimant_ids, provider_ids):
        u = self.claimant_emb(claimant_ids)
        i = self.provider_emb(provider_ids)
        # L2-normalize for cosine-like similarity
        u = F.normalize(u, p=2, dim=1)
        i = F.normalize(i, p=2, dim=1)
        return u, i

