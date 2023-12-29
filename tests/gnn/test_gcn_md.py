import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNMD(nn.Module):
    def __init__(self, in_channels_modality1, in_channels_modality2, out_channels):
        super(GCNMD, self).__init__()
        # Define separate GCN layers for each modality
        self.conv1_modality1 = GCNConv(in_channels_modality1, out_channels)
        self.conv1_modality2 = GCNConv(in_channels_modality2, out_channels)

    def forward(self, data):
        x_modality1, x_modality2, edge_index = data.x_modality1, data.x_modality2, data.edge_index

        # Apply convolutions separately
        x_modality1 = F.relu(self.conv1_modality1(x_modality1, edge_index))
        x_modality2 = F.relu(self.conv1_modality2(x_modality2, edge_index))

        # Combine modalities
        x_combined = x_modality1 + x_modality2  # Example combination, can be modified

        return F.log_softmax(x_combined, dim=1)

def main():
    # Sample graph data
    num_nodes = 10
    in_channels_modality1 = 5
    in_channels_modality2 = 3
    out_channels = 2

    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long).t().contiguous()
    x_modality1 = torch.randn((num_nodes, in_channels_modality1))
    x_modality2 = torch.randn((num_nodes, in_channels_modality2))
    data = Data(x_modality1=x_modality1, x_modality2=x_modality2, edge_index=edge_index)

    # Initialize and apply the GCN-MD model
    model = GCNMD(in_channels_modality1, in_channels_modality2, out_channels)
    output = model(data)

    print("Output from GCN-MD:", output)

if __name__ == "__main__":
    main()
