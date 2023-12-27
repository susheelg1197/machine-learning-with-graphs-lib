import unittest
from ml_wg.graph_generation.erdos_renyi import generate_erdos_renyi
import networkx as nx

class TestErdosRenyiGraph(unittest.TestCase):
    def test_graph_generation(self):
        n = 100
        p = 0.1
        G = generate_erdos_renyi(n, p, seed=42)
        
        # Test number of nodes
        self.assertEqual(len(G.nodes()), n)
        
        # Test graph is connected
        self.assertTrue(nx.is_connected(G))
        
        # Test community structure
        communities = 5
        intra_p = 0.5
        inter_p = 0.01
        G_comm = generate_erdos_renyi(n, p, communities=communities, intra_p=intra_p, inter_p=inter_p, seed=42)
        for c in nx.connected_components(G_comm):
            self.assertTrue(len(c) > 0)  # Each community should have at least one node

        # Test weights
        G_weighted = generate_erdos_renyi(n, p, weights=True, seed=42)
        for u, v, data in G_weighted.edges(data=True):
            self.assertIn('weight', data)

if __name__ == '__main__':
    unittest.main()
