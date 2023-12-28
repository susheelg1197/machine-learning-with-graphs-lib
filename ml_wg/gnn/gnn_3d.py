import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import numpy as np

def create_knn_graph(points, k):
    tree = cKDTree(points)
    edges = tree.query(points, k=k + 1)[1][:, 1:]  # Exclude self-loops
    row = np.repeat(range(len(points)), k)
    col = edges.flatten()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


class GNN3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN3D, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.log_softmax(self.conv2(x, edge_index), dim=1)
        return x
