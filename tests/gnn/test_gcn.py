# tests/gnn/test_gcn.py

import torch
from ml_wg.gnn.gcn import GraphConvolution, GCN

def test_graph_convolution():
    input_feat = 10
    output_feat = 5
    num_nodes = 4

    x = torch.rand(num_nodes, input_feat)
    adj = torch.eye(num_nodes)

    conv = GraphConvolution(input_feat, output_feat)
    assert conv(x, adj).size() == (num_nodes, output_feat), "GraphConvolution output size should match."

def test_gcn():
    nfeat = 10
    nhid = 16
    nclass = 3
    dropout = 0.5
    num_nodes = 4

    x = torch.rand(num_nodes, nfeat)
    adj = torch.eye(num_nodes)

    model = GCN(nfeat, nhid, nclass, dropout)
    output = model(x, adj)
    
    assert output.size() == (num_nodes, nclass), "GCN output size should match the number of nodes and classes."

if __name__ == '__main__':
    test_graph_convolution()
    test_gcn()
    print("Everything passed")
