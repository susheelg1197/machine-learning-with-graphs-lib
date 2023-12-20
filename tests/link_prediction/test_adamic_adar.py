import unittest
import networkx as nx
from ml_wg.link_prediction.adamic_adar import adamic_adar_index

class TestAdamicAdarIndex(unittest.TestCase):
    def test_adamic_adar_index(self):
        G = nx.Graph([(1, 2), (2, 3), (3, 1)])
        index = adamic_adar_index(G, 1, 3)
        self.assertGreaterEqual(index, 0)

if __name__ == '__main__':
    unittest.main()
