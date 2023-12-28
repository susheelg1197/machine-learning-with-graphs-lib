import unittest
import torch
from ml_wg.gnn.gnn_3d import GNN3D, create_knn_graph
from torch_geometric.data import Data

class TestGNN3D(unittest.TestCase):
    def test_gnn_3d_on_sample_data(self):
        # Sample 3D point cloud data
        points = torch.rand((10, 3))  # 10 points in 3D space
        edge_index = create_knn_graph(points.numpy(), k=3)

        # Sample feature data and labels
        x = torch.rand((10, 5))  # 5 features per point
        y = torch.randint(0, 2, (10,))  # Binary labels

        # Create a data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Initialize and apply GNN-3D model
        model = GNN3D(in_channels=5, out_channels=2)
        output = model(data)
        self.assertEqual(output.shape, (10, 2))

if __name__ == '__main__':
    unittest.main()
