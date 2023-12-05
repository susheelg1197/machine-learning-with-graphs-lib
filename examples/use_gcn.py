# examples/use_gcn.py

import torch
from ml_wg.gnn import GCN
from ml_wg.utils import normalize_adjacency  # Assuming you have such a function

def load_data():
    # Placeholder function to load your data
    # In practice, you would load your graph data here
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    labels = torch.randint(0, 3, (10,))  # 10 labels for a 3-class problem
    adjacency_matrix = torch.eye(10)  # Identity matrix as a placeholder
    return node_features, labels, adjacency_matrix

def train_gcn(node_features, labels, adjacency_matrix):
    # Normalize the adjacency matrix
    normalized_adjacency = normalize_adjacency(adjacency_matrix)

    # Create a GCN model
    model = GCN(nfeat=node_features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5)
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
train_gcn(node_features, labels, adjacency_matrix)
