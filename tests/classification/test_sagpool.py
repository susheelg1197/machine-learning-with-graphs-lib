import unittest
import torch
from ml_wg.classification.sagpool import SAGPool
from torch_geometric.data import Data

class TestSAGPool(unittest.TestCase):
    def test_sagpool_forward(self):
        num_features = 3
        num_classes = 2
        model = SAGPool(num_features, num_classes)

        # Example data
        x = torch.randn((4, num_features))
        edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long).t()
        data = Data(x=x, edge_index=edge_index)

        output = model(data)
        self.assertEqual(output.shape[1], num_classes)

if __name__ == '__main__':
    unittest.main()
