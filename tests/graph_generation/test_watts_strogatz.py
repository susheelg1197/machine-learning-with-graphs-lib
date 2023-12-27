import unittest
from ml_wg.graph_generation.watts_strogatz import generate_watts_strogatz
import networkx as nx

class TestWattsStrogatzGraph(unittest.TestCase):
    def test_graph_generation_basic(self):
        n, k, p = 100, 4, 0.1
        G = generate_watts_strogatz(n, k, p)
        self.assertEqual(len(G.nodes()), n)
        self.assertTrue(nx.is_connected(G))

    def test_graph_generation_with_weights(self):
        n, k, p = 50, 6, 0.05
        G = generate_watts_strogatz(n, k, p, add_weights=True)
        for _, _, data in G.edges(data=True):
            self.assertIn('weight', data)
            self.assertTrue(0 <= data['weight'] <= 1)

    def test_graph_generation_with_attributes(self):
        n, k, p = 30, 2, 0.2
        G = generate_watts_strogatz(n, k, p, add_attributes=True)
        for node in G.nodes(data=True):
            self.assertIn('attribute', node[1])
            self.assertTrue(0 <= node[1]['attribute'] <= 1)

if __name__ == '__main__':
    unittest.main()
