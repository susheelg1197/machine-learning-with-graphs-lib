import torch
from ml_wg.gnn.tgn import TemporalGraphNetwork

# Example input dimensions
node_features = 5
edge_features = 3
temporal_features = 4
num_nodes = 10

# Create a TGN model
model = TemporalGraphNetwork(node_features, edge_features, temporal_features)

# Random example input
node_embeddings = torch.rand(num_nodes, node_features)
edge_embeddings = torch.rand(num_nodes, edge_features)
temporal_embeddings = torch.rand(num_nodes, temporal_features)

# Forward pass
output = model(node_embeddings, edge_embeddings, temporal_embeddings)
print("Output from TGN: ", output)
