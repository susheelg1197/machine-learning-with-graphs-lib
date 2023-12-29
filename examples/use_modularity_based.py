import numpy as np
from ml_wg.clustering.modularity_based import ModularityBasedClustering

def main():
    # Example adjacency matrix (you should replace this with your actual graph data)
    adj_matrix = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])

    # Initialize and fit the modularity based clustering
    clustering = ModularityBasedClustering()
    clustering.fit(adj_matrix)

    # Retrieve the clusters
    clusters = clustering.get_clusters()
    print("Detected clusters:", clusters)

if __name__ == "__main__":
    main()
