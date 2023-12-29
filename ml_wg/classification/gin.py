import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(num_features, 16))
        self.conv2 = GINConv(nn.Linear(16, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)
