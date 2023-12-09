# tests/gnn/test_generative.py

import torch
from ml_wg.gnn.generative import GraphGenerativeNN

def test_graph_generative_nn():
    nfeat = 10
    nhid = 16
    nclass = 10
    dropout = 0.5
    num_nodes = 4

    x = torch.rand(num_nodes, nfeat)
    adj = torch.eye(num_nodes)

    model = GraphGenerativeNN(nfeat,nhid, nclass, dropout)
    x_reconstructed = model(x, adj)
    
    assert x_reconstructed.size() == (num_nodes, nfeat), "Output size should match input feature size."

if __name__ == '__main__':
    test_graph_generative_nn()
    print("All generative GNN tests passed")
