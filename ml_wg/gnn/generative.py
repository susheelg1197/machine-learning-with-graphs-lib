# gnn/generative.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GraphConvolution

class GraphGenerativeNN(nn.Module):
    """
    Simple generative model for graphs which aims to reconstruct the adjacency matrix.
    """
    def __init__(self, nfeat, nhid, num_nodes, dropout=0.5):
        super(GraphGenerativeNN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # The output size should be the same as the number of nodes
        self.gc2 = GraphConvolution(nhid, num_nodes)  
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x_reconstructed = torch.sigmoid(self.gc2(x, adj))  # Output shape: [num_nodes, num_nodes]
        return x_reconstructed
