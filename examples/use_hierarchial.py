import torch
from ml_wg.clustering.hierarchical import HierarchicalClustering

def create_example_graph(num_nodes=10):
    # Generate a random symmetric adjacency matrix with self-loops
    adj_matrix = torch.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2  # Make it symmetric
    adj_matrix.fill_diagonal_(1)  # Corrected line to add self-loops
    return adj_matrix


def main():
    num_nodes = 10
    adj_matrix = create_example_graph(num_nodes)

    hc = HierarchicalClustering()
    hc.fit(adj_matrix)
    clusters = hc.get_clusters(num_clusters=3)

    print("Clusters:", clusters)

if __name__ == "__main__":
    main()
