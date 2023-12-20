import unittest
import torch
from torch_geometric.data import Data
from ml_wg.gnn.unet import GraphUNet

class TestGraphUNet(unittest.TestCase):
    def test_graph_unet_forward(self):
        in_channels = 3
        hidden_channels = 16
        out_channels = 2
        num_nodes = 4
        model = GraphUNet(in_channels, hidden_channels, out_channels)

        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long).t()
        data = Data(x=x, edge_index=edge_index)
        batch = torch.tensor([0, 0, 1, 1])

        output = model(data.x, data.edge_index, batch)
        self.assertEqual(output.size(0), 2)  # Assuming 2 graphs in the batch

if __name__ == '__main__':
    unittest.main()
