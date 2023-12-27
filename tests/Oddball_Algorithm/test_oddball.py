import unittest
from ml_wg.OddBall_Algorithm.oddball import oddball, compute_egonet_features
import networkx as nx
import numpy as np

class TestOddBall(unittest.TestCase):
    def setUp(self):
        self.G = nx.fast_gnp_random_graph(10, 0.5, seed=42)
        for u, v in self.G.edges():
            self.G.edges[u, v]['weight'] = np.random.rand()

    def test_odd_ball_detection(self):
        anomalies = oddball(self.G)
        # Test based on the known properties of the graph
        # This is a simplistic check, more robust tests might be needed
        self.assertIsInstance(anomalies, list)
        self.assertTrue(all(isinstance(node, int) for node in anomalies))

    def test_egonet_features(self):
        node = np.random.choice(self.G.nodes())
        egonet = nx.ego_graph(self.G, node)
        features = compute_egonet_features(egonet)
        self.assertIn('average_degree', features)
        self.assertIn('skewness_degree', features)
        self.assertIn('kurtosis_degree', features)

if __name__ == '__main__':
    unittest.main()
