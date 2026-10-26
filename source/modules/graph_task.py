import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool
from source.main_layers.kuramoto_layer import KuramotoLayer
from source.main_layers.readout_layers import ReadoutGNN

class KuramotoGNN(nn.Module):
    def __init__(self, 
                    in_dim,
                    osc_dim, 
                    final_dim,
                    L,
                    T, 
                    hidden_dim,
                    message_passing_type = "attn",
                    num_head = 8,
                    gamma = 1.0,
                    edge_dim = 0,
                    readout_type = "gat",
                    ):
        super(KuramotoGNN, self).__init__()

        self.osc_dim = osc_dim
        self.L = L
        self.T = T
        self.hidden_dim = hidden_dim
        self.message_passing_type = message_passing_type
        self.edge_dim = edge_dim

        self.gamma = gamma

        self.embedding = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)

        self.layers = nn.ModuleList()

        for i in range(L):
            kuramoto_layer = KuramotoLayer(
                osc_dim=osc_dim,
                T = T,
                hidden_dim=hidden_dim,
                message_passing_type=message_passing_type,
                num_head=num_head,
                gamma=gamma,
                edge_dim=edge_dim
            )

            # if readout_type == "gat":
            readout_layer = ReadoutGNN(in_dim=hidden_dim, out_dim=hidden_dim, osc_dim=osc_dim, readout_type=readout_type)
            # else:
            #     raise NotImplementedError

            self.layers.append(nn.ModuleList([kuramoto_layer, readout_layer]))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, final_dim)
        )

    def forward(self, x: torch.Tensor, 
                    edge_index : torch.Tensor, 
                    edge_attr : torch.Tensor = None,
                    batch = None,
                    ):

        reference = self.embedding(x)
        osc =  torch.rand_like(reference)

        track_osc = [osc]
        for klayer, readout_layer in self.layers:
            list_osc = klayer(x = osc, reference = reference, edge_index=edge_index, edge_attr = edge_attr)
            osc = list_osc[-1]
            reference = readout_layer(osc, edge_index)
            track_osc.append(list_osc)

        grap_repr = global_add_pool(reference, batch=batch)
        output = self.out_layer(grap_repr)
        return output, reference, track_osc



if __name__ == "__main__":
    import rdkit
    import random
    import numpy as np
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    from torch_geometric.utils.smiles import from_smiles
    g = from_smiles("CCCCCC", with_hydrogen=True)
    g.x = g.x.float() @ torch.rand((9, 32))
    print(g)
    model = KuramotoGNN(
        in_dim=32, osc_dim=4, final_dim=1, L=2, T=5, hidden_dim=64,
        message_passing_type = "attn", num_head=4, gamma=1.0, edge_dim=0, readout_type="gat"
    )

    output = model(g.x, g.edge_index)


    print(output[0])
    print(output[0].shape)
    for i in output[1]:
        print(len(i))
        print([j.shape for j in i])


        


