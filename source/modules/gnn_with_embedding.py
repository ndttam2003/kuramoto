import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool
from source.main_layers.kuramoto_layer import KuramotoLayer
from source.main_layers.readout_layers import ReadoutGNN
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from .helper_layers import ResidualFF, GAT

class NodeGNN(nn.Module):
    def __init__(self, 
                    in_dim,
                    osc_dim, 
                    final_dim,
                    L,
                    T, 
                    hidden_dim,
                    n_gnn_layers,
                    gnn_layer_type = "gcn",
                    message_passing_type = "attn",
                    num_head = 8,
                    gamma = 1.0,
                    edge_dim = 0,
                    readout_type = "gat",
                    ):
        super(NodeGNN, self).__init__()

        self.osc_dim = osc_dim
        self.L = L
        self.T = T
        self.hidden_dim = hidden_dim
        self.message_passing_type = message_passing_type
        self.edge_dim = edge_dim

        self.n_gnn_layers = n_gnn_layers

        self.gamma = gamma

        self.node_embedding = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)

        self.kuramoto_layers = nn.ModuleList()

        for _ in range(L):
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

            self.kuramoto_layers.append(nn.ModuleList([kuramoto_layer, readout_layer]))


        self.gnn_layers = nn.ModuleList()

        for _ in range(n_gnn_layers):
            if gnn_layer_type == "gcn":
                gnn_l = GCNConv(hidden_dim, hidden_dim)
            elif gnn_layer_type == "gat":
                assert hidden_dim // num_head != 0
                gnn_l = GAT(hidden_dim, hidden_dim, num_head=num_head)
            else:
                raise NotImplementedError

            act_fn = nn.SiLU()

            # f_c_h = nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(hidden_dim*2, hidden_dim),
            #     ResidualFF(hidden_dim, hidden_dim),
            # )

            self.gnn_layers.append(nn.ModuleList([gnn_l, act_fn]))
            


        self.out_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(4*hidden_dim, final_dim)
        )

    def kuramoto(self, osc: torch.Tensor, 
                    reference: torch.Tensor,
                    edge_index : torch.Tensor, 
                    edge_attr : torch.Tensor = None,
                    ):

        track_osc = [osc]
        for klayer, readout_layer in self.kuramoto_layers:
            list_osc = klayer(x = osc, reference = reference, edge_index=edge_index, edge_attr = edge_attr)
            osc = list_osc[-1]
            reference = readout_layer(osc, edge_index)
            track_osc.append(list_osc)

        return reference, track_osc

    def forward(self, x: torch.Tensor, 
                    edge_index : torch.Tensor, 
                    edge_attr : torch.Tensor = None,
                    batch = None,
                    ):

        embedd = self.node_embedding(x)
        reference = embedd
        h = embedd
        osc =  torch.rand_like(reference)

        track_osc = []
        track_reference = []
        reference, track = self.kuramoto(osc, reference, edge_index, edge_attr)
        track_osc.append(track)
        track_reference.append(reference)
        for gnn_l, act_fn in self.gnn_layers:

            h = h + reference
            h = act_fn(h)
            h = gnn_l(h, edge_index)

            # reference = f_c_h(torch.cat((h, reference), dim=-1))
        
        out = h

        grap_repr = global_add_pool(out, batch=batch)
        output = self.out_layer(grap_repr)
        return output, track_reference, track_osc


class EdgeGNN(nn.Module):
    def __init__(self, 
                    in_dim,
                    osc_dim, 
                    final_dim,
                    L,
                    T, 
                    hidden_dim,
                    n_gnn_layers,
                    gnn_layer_type = "gcn",
                    message_passing_type = "attn",
                    num_head = 8,
                    gamma = 1.0,
                    edge_dim = 0,
                    readout_type = "gat",
                    ):
        super(EdgeGNN, self).__init__()

        self.osc_dim = osc_dim
        self.L = L
        self.T = T
        self.hidden_dim = hidden_dim
        self.message_passing_type = message_passing_type
        self.edge_dim = edge_dim

        self.n_gnn_layers = n_gnn_layers

        self.gamma = gamma

        self.node_embedding = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)

        self.kuramoto_layers = nn.ModuleList()

        for _ in range(L):
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

            self.kuramoto_layers.append(nn.ModuleList([kuramoto_layer, readout_layer]))


        self.gnn_layers = nn.ModuleList()
        self.gnn_layer_type = gnn_layer_type

        # if gnn_layer_type == "gcn":
        #     self.lin_weight =  nn.Sequential(
        #                 nn.SiLU(),
        #                 nn.Linear(hidden_dim, hidden_dim*4),
        #                 nn.SiLU(),
        #                 nn.Linear(4*hidden_dim, 1)
        #             )

        for _ in range(n_gnn_layers):
            if gnn_layer_type == "gcn":
                gnn_l = GCNConv(hidden_dim, hidden_dim)
            elif gnn_layer_type == "gat":
                assert hidden_dim // num_head != 0
                gnn_l = GAT(hidden_dim, hidden_dim, num_head=num_head, edge_dim=hidden_dim)
            else:
                raise NotImplementedError

            act_fn = nn.SiLU()

            # f_c_h = nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(hidden_dim*2, hidden_dim),
            #     ResidualFF(hidden_dim, hidden_dim),
            # )

            self.gnn_layers.append(nn.ModuleList([gnn_l, act_fn]))
            


        self.out_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(4*hidden_dim, final_dim)
        )

    def kuramoto(self, osc: torch.Tensor, 
                        reference: torch.Tensor,
                        edge_index : torch.Tensor, 
                        edge_attr : torch.Tensor = None,
                        ):
    
        track_osc = [osc]
        for klayer, readout_layer in self.kuramoto_layers:
            list_osc = klayer(x = osc, reference = reference, edge_index=edge_index, edge_attr = edge_attr)
            osc = list_osc[-1]
            reference = readout_layer(osc, edge_index)
            track_osc.append(list_osc)

        return reference, track_osc

    def get_edge(self,  reference: torch.Tensor, 
                    edge_index : torch.Tensor):
        src, target = edge_index
        if self.gnn_layer_type == "gcn":
            x = F.normalize(reference, dim=-1)
            x_i = x[src]
            x_j = x[target]

            return torch.sum(x_i*x_j, dim=-1, keepdim=True)
        elif self.gnn_layer_type == "gat":
            return reference[target] - reference[src]
        else:
            raise NotImplementedError


    def forward(self, x: torch.Tensor, 
                    edge_index : torch.Tensor, 
                    edge_attr : torch.Tensor = None,
                    batch = None,
                    ):

        embedd = self.node_embedding(x)
        reference = embedd
        h = embedd
        osc =  torch.rand_like(reference)

        track_osc = []
        track_reference = []
        reference, track = self.kuramoto(osc, reference, edge_index, edge_attr)
        track_osc.append(track)
        track_reference.append(reference)
        for gnn_l, act_fn in self.gnn_layers:

            weigh = self.get_edge(reference=reference, edge_index=edge_index)
            # h = h
            h = act_fn(h)
            h = gnn_l(h, edge_index, weigh)

            # reference = f_c_h(torch.cat((h, reference), dim=-1))
        
        out = h

        grap_repr = global_add_pool(out, batch=batch)
        output = self.out_layer(grap_repr)
        return output, track_reference, track_osc


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
    model = EdgeGNN(
        in_dim=32,
        osc_dim=4,
        final_dim=1,
        L=2,
        T=5,
        hidden_dim=64,
        n_gnn_layers = 3,
        gnn_layer_type = "gat",
        message_passing_type = "attn",
        num_head=4,
        gamma=1.0,
        edge_dim=0,
        readout_type="gat"
    )

    print(model)

    output, track_reference, track_osc = model(g.x, g.edge_index)


    print(output)
    print(output.shape)
    # for i in output[1]:
    #     print(len(i))
    #     print([j.shape for j in i])


        



