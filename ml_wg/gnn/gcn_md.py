import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNMD(nn.Module):
    def __init__(self, in_channels_modality1, in_channels_modality2, out_channels):
        super(GCNMD, self).__init__()
        self.conv1_modality1 = GCNConv(in_channels_modality1, 16)
        self.conv1_modality2 = GCNConv(in_channels_modality2, 16)
        self.conv2 = GCNConv(32, out_channels)  # Merging both modalities

    def forward(self, data):
        x_modality1, x_modality2, edge_index = data.x_modality1, data.x_modality2, data.edge_index

        # First modality
        x_modality1 = F.relu(self.conv1_modality1(x_modality1, edge_index))

        # Second modality
        x_modality2 = F.relu(self.conv1_modality2(x_modality2, edge_index))

        # Concatenate features from both modalities
        x = torch.cat([x_modality1, x_modality2], dim=1)

        # Further processing with combined features
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
