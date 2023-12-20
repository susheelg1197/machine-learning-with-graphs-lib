import unittest
import torch
from ml_wg.gnn.graph_transformer import GraphTransformer

class TestGraphTransformer(unittest.TestCase):
    def test_graph_transformer_forward(self):
        n_layers = 2
        in_features = 4  # Adjusted to be divisible by num_heads
        out_features = 10
        num_heads = 2
        num_nodes = 4

        model = GraphTransformer(n_layers, in_features, out_features, num_heads)
        x = torch.rand(num_nodes, in_features)

        output = model(x)
        self.assertEqual(output.shape, (num_nodes, in_features))

if __name__ == '__main__':
    unittest.main()
