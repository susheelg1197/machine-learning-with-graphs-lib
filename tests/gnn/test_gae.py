# tests/gnn/test_gae.py

import torch
from ml_wg.gnn.gae import GraphAutoencoder

def test_graph_autoencoder():
    nfeat = 10
    nhid = 16
    nclass = 5
    dropout = 0.5
    num_nodes = 4

    model = GraphAutoencoder(nfeat, nhid, nclass, dropout)
    x = torch.rand(num_nodes, nfeat)
    adj = torch.eye(num_nodes)

    A_pred = model(x, adj)
    assert A_pred.size() == (num_nodes, num_nodes), "GAE output size should match."

if __name__ == '__main__':
    test_graph_autoencoder()
    print("All GAE tests passed")
