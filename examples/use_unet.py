import torch
from torch_geometric.data import Data
from ml_wg.gnn.unet import GraphUNet

# Graph parameters
in_channels = 3
hidden_channels = 16
out_channels = 2
num_nodes = 4

# Create a Graph U-Net model
model = GraphUNet(in_channels, hidden_channels, out_channels)

# Example input
x = torch.randn(num_nodes, in_channels)
edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long).t()
batch = torch.tensor([0, 0, 1, 1])  # Assuming two graphs
data = Data(x=x, edge_index=edge_index)

output = model(data.x, data.edge_index, batch)
print("Output from Graph U-Net: ", output)
