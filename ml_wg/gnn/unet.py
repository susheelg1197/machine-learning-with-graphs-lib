import torch
import torch.nn as nn
import torch.nn.functional as F  

from torch_geometric.nn import TopKPooling, GCNConv, global_mean_pool

class GraphUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphUNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Decoder part
        self.deconv1 = GCNConv(out_channels, hidden_channels)
        self.deconv2 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))

        # Decoder
        x = F.relu(self.deconv1(x, edge_index))
        x = self.deconv2(x, edge_index)
        return global_mean_pool(x, batch)
