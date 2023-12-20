import unittest
import torch
from torch_geometric.data import Data
from ml_wg.gnn.hgn import HGN

class TestHGN(unittest.TestCase):
    def test_hgn_forward(self):
        node_dim = 5
        edge_dim = 3
        num_nodes = 4
        model = HGN(node_dim, edge_dim)

        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long).t()
        edge_attr = torch.randn(edge_index.size(1), edge_dim)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        output = model(data)
        self.assertEqual(output.size(), (num_nodes, node_dim))

if __name__ == '__main__':
    unittest.main()
