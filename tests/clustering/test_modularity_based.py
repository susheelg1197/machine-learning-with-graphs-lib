import numpy as np
import unittest
from ml_wg.clustering.modularity_based import ModularityBasedClustering

class TestModularityBasedClustering(unittest.TestCase):

    def test_modularity_clustering(self):
        adj_matrix = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        
        modularity_clustering = ModularityBasedClustering()
        modularity_clustering.fit(adj_matrix)
        clusters = modularity_clustering.get_clusters()

        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)

if __name__ == '__main__':
    unittest.main()
