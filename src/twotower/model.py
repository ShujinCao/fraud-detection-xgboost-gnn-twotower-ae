import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, n_claimants, n_providers, embed_dim):
        super().__init__()
        self.claimant_emb = nn.Embedding(n_claimants + 1, embed_dim)
        self.provider_emb = nn.Embedding(n_providers + 1, embed_dim)

    def forward(self, claimant_ids, provider_ids):
        u = self.claimant_emb(claimant_ids)
        i = self.provider_emb(provider_ids)
        scores = (u * i).sum(dim=1)
        return scores, u, i

