import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ResidualFF(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualFF, self).__init__()
        hidden_dim = in_dim * 4

        # act -> NN ==> smooth gradient through skip connection
        self.ff = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim)
                )

    def forward(self, x):
        return x + self.ff(x)

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, edge_dim = None):
        super(GAT, self).__init__()
        self.num_head = num_head

        self.gat = GATv2Conv(in_channels=in_dim, out_channels=out_dim, heads= num_head, edge_dim=edge_dim)
        self.lin = nn.Linear(out_dim * num_head, out_dim, bias=False)

    def forward(self, x, edge_index, edge_attr = None):
        x = self.gat(x, edge_index, edge_attr = edge_attr)
        return self.lin(x)