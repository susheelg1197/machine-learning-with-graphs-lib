import numpy as np
from ml_wg.clustering.spectral import SpectralClustering

def create_example_graph(num_nodes=10):
    # Generate a random symmetric adjacency matrix with zero diagonal
    adj_matrix = np.random.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def main():
    num_nodes = 10
    adj_matrix = create_example_graph(num_nodes)

    sc = SpectralClustering(n_clusters=3)
    clusters = sc.fit_predict(adj_matrix)

    print("Clusters:", clusters)

if __name__ == "__main__":
    main()
