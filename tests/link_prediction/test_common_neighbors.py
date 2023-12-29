import unittest
import networkx as nx
from ml_wg.link_prediction.common_neighbors import common_neighbors

class TestCommonNeighbors(unittest.TestCase):
    def test_common_neighbors(self):
        G = nx.Graph([(1, 2), (2, 3), (1, 3)])
        count = common_neighbors(G, 1, 3)
        self.assertEqual(count, 1)

if __name__ == '__main__':
    unittest.main()
