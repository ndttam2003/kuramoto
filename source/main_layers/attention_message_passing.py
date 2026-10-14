import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.utils import add_self_loops
import torch_scatter
import einops
import math


class AttentionMessagePassing(nn.Module):
    def __init__(self, hidden_dim, num_head, edge_dim = 0):
        super(AttentionMessagePassing, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        print(hidden_dim, num_head)
        assert hidden_dim // num_head != 0
        self.W_qkv = nn.Linear(hidden_dim + edge_dim, hidden_dim * 3, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)


    def without_edge_attr(self,  
                          x: torch.Tensor, 
                          edge_index : torch.Tensor,):
        edge_index_self_loop, _ = add_self_loops(edge_index=edge_index)
        src, target =  edge_index_self_loop
        Q, K, V = self.W_qkv(x).chunk(3, dim = -1)
        # print(Q.shape, K.shape, V.shape)
        Q_i = einops.rearrange(Q[target], "b (nh d) -> b nh d", nh = self.num_head)
        K_j = einops.rearrange(K[src], "b (nh d) -> b nh d", nh = self.num_head)
        V_j = einops.rearrange(V[src], "b (nh d) -> b nh d", nh = self.num_head)

        score = torch.sum(Q_i * K_j, dim = -1, keepdim=True) / math.sqrt(self.hidden_dim // self.num_head) # [B, nh, 1]
        # attn = softmax(src = score.transpose(0, 1), index=target, dim=1).transpose(0, 1)# [B, nh, 1]
        attn = softmax(src = score, index=target, dim=0)# [B, nh, 1]

        new_x = torch_scatter.scatter(src= attn * V_j, index=target, dim=0) 
        new_x = einops.rearrange(new_x, "b nh d -> b (nh d)", nh = self.num_head)

        out = self.W_o(new_x)
        return out

    def forward(self, 
                x: torch.Tensor, 
                edge_index : torch.Tensor, 
                edge_attr : torch.Tensor = None):
        if edge_attr == None:
            return self.without_edge_attr(x, edge_index)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import rdkit
    import random
    import numpy as np
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    from torch_geometric.utils.smiles import from_smiles
    g = from_smiles("CCCCCC", with_hydrogen=True)
    g.x = g.x.float() @ torch.rand((9, 16))
    print(g)
    model = AttentionMessagePassing(16, 2)

    output = model(g.x, g.edge_index)


    print(output)
    print(output.shape)




            



