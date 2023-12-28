import torch
from ml_wg.gnn.gnn_3d import GNN3D, create_knn_graph
from torch_geometric.data import Data

def main():
    # Sample 3D point cloud data
    points = torch.rand((10, 3))  # 10 points in 3D space
    edge_index = create_knn_graph(points.numpy(), k=3)

    # Sample feature data
    x = torch.rand((10, 5))  # 5 features per point

    # Create a data object
    data = Data(x=x, edge_index=edge_index)

    # Initialize and apply GNN-3D model
    model = GNN3D(in_channels=5, out_channels=2)
    output = model(data)

    print("Output from GNN-3D:", output)

if __name__ == "__main__":
    main()
