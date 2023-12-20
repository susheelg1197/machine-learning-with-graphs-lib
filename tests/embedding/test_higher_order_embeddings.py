import unittest
import networkx as nx
from ml_wg.embedding.higher_order_embeddings import higher_order_embeddings

class TestHigherOrderEmbeddings(unittest.TestCase):

    def test_higher_order_embeddings(self):
        G = nx.complete_graph(5)
        embeddings = higher_order_embeddings(G, motif_size=3)

        # Test if embeddings are generated for each node
        for node in G.nodes():
            self.assertIn(node, embeddings)
            self.assertEqual(len(embeddings[node]), 128)

if __name__ == '__main__':
    unittest.main()
