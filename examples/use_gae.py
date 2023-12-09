# examples/use_gae.py

import torch
from ml_wg.gnn import GraphAutoencoder
from ml_wg.utils import normalize_adjacency
import torch.nn.functional as F


def load_data():
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    adjacency_matrix = torch.eye(10)  # Identity matrix as a placeholder
    return node_features, adjacency_matrix

def train_gae(node_features, adjacency_matrix):
    model = GraphAutoencoder(nfeat=node_features.shape[1], nhid=16, nclass=5, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        A_pred = model(node_features, adjacency_matrix)
        loss = F.binary_cross_entropy(A_pred, adjacency_matrix)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: loss = {loss.item()}')

node_features, adjacency_matrix = load_data()
train_gae(node_features, adjacency_matrix)
