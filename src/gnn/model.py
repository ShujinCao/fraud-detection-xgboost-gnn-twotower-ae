import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj_lists):
        """
        x: (N, D) node embeddings
        adj_lists: list of neighbor lists; adj_lists[v] = list of neighbors of v
        """
        device = x.device
        out = torch.zeros_like(x)
        for v in range(x.size(0)):
            neighs = adj_lists[v]
            if len(neighs) == 0:
                neigh_agg = x[v]
            else:
                neigh_agg = x[neighs].mean(dim=0)
            h_v = torch.cat([x[v], neigh_agg], dim=-1)
            out[v] = self.linear(h_v)
        return F.relu(out)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            in_d = dims[i]
            out_d = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(GraphSAGELayer(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj_lists):
        for layer in self.layers:
            x = layer(x, adj_lists)
        return x

