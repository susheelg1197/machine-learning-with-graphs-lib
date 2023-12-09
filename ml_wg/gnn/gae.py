# gnn/gae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GraphConvolution


class GraphAutoencoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GraphAutoencoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def encode(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        return self.gc2(hidden1, adj)

    def forward(self, x, adj):
        z = self.encode(x, adj)
        A_pred = torch.sigmoid(torch.mm(z, z.t()))
        return A_pred
