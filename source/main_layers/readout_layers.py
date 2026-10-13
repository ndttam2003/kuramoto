import torch
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
import einops

class ReadoutGNN(nn.Module):
    def __init__(self, 
                    in_dim,
                    out_dim,
                    osc_dim, 
                    readout_type = "gat",
                    ):
        super(ReadoutGNN, self).__init__()

        self.osc_dim = osc_dim

        if readout_type == "gat":
            self.gnn = GATv2Conv(in_channels=in_dim,
                                 out_channels= out_dim * osc_dim
                                 )
        else:
            raise NotImplementedError

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = einops.rearrange(x, "b (n d) -> b n d", d = self.osc_dim)
        x = norm(x, dim = -1)
        return x + self.bias.unsqueeze(0)