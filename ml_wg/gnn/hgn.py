import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class HamiltonianLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(HamiltonianLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Construct messages as a function of node and edge features.
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

class HGN(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(HGN, self).__init__()
        self.hamiltonian_layer = HamiltonianLayer(node_dim, edge_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.hamiltonian_layer(x, edge_index, edge_attr)
        return x
