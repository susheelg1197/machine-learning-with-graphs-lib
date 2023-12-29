import unittest
import torch
from ml_wg.gnn.tgn import TemporalGraphNetwork

class TestTemporalGraphNetwork(unittest.TestCase):
    def test_tgn_forward(self):
        node_features = 5
        edge_features = 3
        temporal_features = 4
        num_nodes = 10

        model = TemporalGraphNetwork(node_features, edge_features, temporal_features)
        node_embeddings = torch.rand(num_nodes, node_features)
        edge_embeddings = torch.rand(num_nodes, edge_features)
        temporal_embeddings = torch.rand(num_nodes, temporal_features)

        output = model(node_embeddings, edge_embeddings, temporal_embeddings)
        self.assertEqual(output.shape, (num_nodes, 128))

if __name__ == '__main__':
    unittest.main()
