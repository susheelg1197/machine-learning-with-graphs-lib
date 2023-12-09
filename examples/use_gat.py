# examples/use_gat.py

import torch
from ml_wg.gnn import GAT
from ml_wg.utils import normalize_adjacency

def load_data():
    # Placeholder function to load your data
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    labels = torch.randint(0, 2, (10,))  # 10 labels for a binary classification problem
    adjacency_matrix = torch.eye(10)  # Identity matrix as a placeholder
    return node_features, labels, adjacency_matrix

def train_gat(node_features, labels, adjacency_matrix):
    # Normalize the adjacency matrix
    normalized_adjacency = normalize_adjacency(adjacency_matrix)

    # Create a GAT model
    model = GAT(nfeat=node_features.shape[1], nhid=8, nclass=2, dropout=0.6, alpha=0.2, nheads=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model (simplified example)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(node_features, normalized_adjacency)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: loss = {loss.item()}')

# Load your data
node_features, labels, adjacency_matrix = load_data()

# Train your model
train_gat(node_features, labels, adjacency_matrix)
