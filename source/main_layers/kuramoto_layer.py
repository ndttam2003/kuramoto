import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from .attention_message_passing import AttentionMessagePassing
import einops

class Omega(nn.Module):
    def __init__(self, hidden_dim):
        super(Omega, self).__init__()
        self.omega = nn.Parameter((1.0 / np.sqrt(2)) * torch.ones(hidden_dim // 2, 2))

    def forward(self, x: torch.Tensor):
        num_nodes, hidden_dim = x.shape
        x = x.reshape((num_nodes, hidden_dim // 2, 2)) # [num nodes, hidden_dim // 2, 2]
        omega : torch.Tensor = LA.norm(self.omega, dim = 1) # [hidden_dim // 2]
        omega = omega.unsqueeze(0)# [1, hidden_dim // 2]

        natural_freq = torch.stack([omega*x[:,:, 1], -omega *x[:,:,0]], dim=-1)
        natural_freq = natural_freq.reshape(num_nodes, hidden_dim)

        return natural_freq

class KuramotoLayer(nn.Module):
    def __init__(self, 
                 osc_dim, 
                 T, 
                 hidden_dim,
                 message_passing_type = "attn",
                 num_head = 8,
                 gamma = 1.0,
                 edge_dim = 0,
                 ):
        super(KuramotoLayer, self).__init__()

        self.osc_dim = osc_dim
        self.T = T
        self.hidden_dim = hidden_dim
        self.message_passing_type = message_passing_type
        self.edge_dim = edge_dim

        self.gamma = gamma

        self.frequency = Omega(hidden_dim=hidden_dim)

        if message_passing_type == "attn":
            self.message_passing = AttentionMessagePassing(
                hidden_dim=hidden_dim,
                num_head=num_head,
                edge_dim=edge_dim
            )

        self.norm_reference = nn.GroupNorm(hidden_dim // osc_dim, hidden_dim)

    def project_on_tangent(self, x: torch.Tensor, y: torch.Tensor,):
        x_osc = einops.rearrange(x, "b (n d) -> b n d", d=self.osc_dim)
        y_osc = einops.rearrange(y, "b (n d) -> b n d", d=self.osc_dim)

        dot_product = torch.sum(x_osc * y_osc, dim=-1, keepdim=True)
        tangent_osc = y_osc - x_osc * dot_product

        tangent = einops.rearrange(tangent_osc, "b n d -> b (n d)", d=self.osc_dim)
        return tangent

    def normalize_osc(self, x: torch.Tensor):
        x = einops.rearrange(x, "b (n d) -> b n d", d=self.osc_dim)
        x = F.normalize(x, dim=-1)
        x = einops.rearrange(x, "b n d -> b (n d)", d=self.osc_dim)
        return x

    def forward(self, 
                x: torch.Tensor, 
                reference: torch.Tensor, 
                edge_index : torch.Tensor, 
                edge_attr : torch.Tensor = None):

        c = self.norm_reference(reference)
        x = self.normalize_osc(x)
        xs = []
        for t in range(self.T):
            frequency = self.frequency(x)
            message = self.message_passing(x = x, edge_index = edge_index, edge_attr = edge_attr)
            projection = self.project_on_tangent(x=x, y = c + message)
            delta_x = frequency + projection
            new_x = x + self.gamma * delta_x
            x = self.normalize_osc(new_x)
            xs.append(x)

        return xs

        







 

