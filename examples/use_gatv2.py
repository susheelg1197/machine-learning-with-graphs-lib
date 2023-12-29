import torch
from gatv2 import GATv2

# Graph parameters
num_nodes = 10
in_features = 5
nhid = 8
nout = 2
nheads = 2

# Create a GATv2 model
model = GATv2(nfeat=in_features, nhid=nhid, nout=nout, nheads=nheads)

# Example input (random data for demonstration)
x = torch.rand((num_nodes, in_features))
adj = torch.randint(0, 2, (num_nodes, num_nodes))

# Forward pass
output = model(x, adj)
print("Output from GATv2: ", output)
