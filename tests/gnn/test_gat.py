# tests/gnn/test_gat.py

import torch
from ml_wg.gnn.gat import GraphAttentionLayer, GAT

def test_graph_attention_layer():
    input_feat = 8
    output_feat = 4
    num_nodes = 3
    dropout = 0.6
    alpha = 0.2
    concat = True

    x = torch.rand(num_nodes, input_feat)
    adj = torch.eye(num_nodes)  # Simple adjacency matrix with self-loops

    gat_layer = GraphAttentionLayer(input_feat, output_feat, dropout, alpha, concat)
    assert gat_layer(x, adj).size() == (num_nodes, output_feat), "GraphAttentionLayer output size should match."

def test_gat():
    nfeat = 8
    nhid = 4
    nclass = 2
    dropout = 0.6
    alpha = 0.2
    nheads = 2

    num_nodes = 3

    x = torch.rand(num_nodes, nfeat)
    adj = torch.eye(num_nodes)

    gat_model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)
    output = gat_model(x, adj)
    
    assert output.size() == (num_nodes, nclass), "GAT output size should match the number of nodes and classes."

if __name__ == '__main__':
    test_graph_attention_layer()
    test_gat()
    print("All GAT tests passed")
