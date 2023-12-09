# examples/use_generative.py

import torch
from ml_wg.gnn import GraphGenerativeNN
import torch.nn.functional as F


def load_data():
    # Placeholder function to load your data
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    adjacency_matrix = torch.eye(10)  # Identity matrix as a placeholder
    return node_features, adjacency_matrix

def train_generative_gnn(node_features, adjacency_matrix):
    num_nodes = adjacency_matrix.size(0)
    model = GraphGenerativeNN(nfeat=node_features.shape[1], nhid=16, nclass=num_nodes, dropout=0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        x_reconstructed = model(node_features, adjacency_matrix)
        loss = F.mse_loss(x_reconstructed, adjacency_matrix)  # Assuming reconstruction loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: loss = {loss.item()}')

node_features, adjacency_matrix = load_data()
train_generative_gnn(node_features, adjacency_matrix)
