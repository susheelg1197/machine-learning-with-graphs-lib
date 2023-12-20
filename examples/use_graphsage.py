import torch
from ml_wg.gnn import GraphSAGE
from ml_wg.utils import load_graph_data  # Assuming you have a function to load graph data

def load_data():
    # Placeholder function to load your data
    # Replace this with actual data loading logic
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    labels = torch.randint(0, 3, (10,))  # 10 labels for a 3-class problem
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=torch.long).t()  # Sample edges
    return node_features, labels, edge_index

def train_graphsage(node_features, labels, edge_index):
    # Create a GraphSAGE model
    model = GraphSAGE(in_channels=node_features.shape[1], hidden_channels=16, out_channels=labels.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model (simplified example)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(node_features, edge_index)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: loss = {loss.item()}')

# Load your data
node_features, labels, edge_index = load_data()

# Train your model
train_graphsage(node_features, labels, edge_index)
