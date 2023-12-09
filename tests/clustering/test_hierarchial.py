# tests/clustering/test_hierarchical.py

import torch
from ml_wg.clustering.hierarchical import HierarchicalClustering

def test_hierarchical_clustering():
    num_nodes = 5
    adj_matrix = torch.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2  # Make it symmetric
    adj_matrix.fill_diagonal_(1)  # Add self-loops

    hc = HierarchicalClustering()
    hc.fit(adj_matrix)
    clusters = hc.get_clusters(num_clusters=2)

    assert len(clusters) == num_nodes, "Number of clusters should match number of nodes."

if __name__ == "__main__":
    test_hierarchical_clustering()
    print("Hierarchical clustering tests passed.")
