import torch
from torch_geometric.datasets import Planetoid
from ml_wg.gnn.graphsage import GraphSAGE

def test_graphsage():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    model = GraphSAGE(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes)
    data = dataset[0]
    out = model(data.x, data.edge_index)
    assert out.shape[1] == dataset.num_classes, "Output dimensions don't match the number of classes."

if __name__ == "__main__":
    test_graphsage()
