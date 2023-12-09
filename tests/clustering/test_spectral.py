import numpy as np
from ml_wg.clustering.spectral import SpectralClustering

def test_spectral_clustering():
    # Example adjacency matrix (symmetric and zero diagonal)
    adj_matrix = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ])

    sc = SpectralClustering(n_clusters=2)
    labels = sc.fit_predict(adj_matrix)
    assert len(labels) == 4, "Number of labels should be equal to the number of nodes."

if __name__ == '__main__':
    test_spectral_clustering()
    print("Spectral clustering tests passed.")
