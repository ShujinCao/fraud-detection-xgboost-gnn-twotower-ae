import torch
import torch.nn as nn

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x: (N, D)
        adj: (N, N) sparse or dense adjacency (normalized)
        """
        x = torch.sparse.mm(adj, x)  # aggregate neighbors
        x = self.linear(x)
        return torch.relu(x)

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = SimpleGCNLayer(in_dim, hidden_dim)
        self.layer2 = SimpleGCNLayer(hidden_dim, out_dim)

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h

