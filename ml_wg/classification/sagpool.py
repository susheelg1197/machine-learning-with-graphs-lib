import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, GCNConv

class SAGPool(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SAGPool, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.pool = SAGPooling(16, ratio=0.5)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)

