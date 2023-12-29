import torch
from ml_wg.gnn.graph_transformer import GraphTransformer

# Graph parameters
n_layers = 2
in_features = 5
out_features = 10
num_heads = 2
num_nodes = 4

# Create a Graph Transformer model
model = GraphTransformer(n_layers, in_features, out_features, num_heads)

# Example input (random data for demonstration)
x = torch.rand(num_nodes, in_features)

# Forward pass
output = model(x)
print("Output from Graph Transformer: ", output)
