import unittest
import networkx as nx
from ml_wg.embedding.metapath2vec import train_metapath2vec

class TestMetapath2Vec(unittest.TestCase):

    def test_metapath2vec(self):
        G = nx.Graph()
        G.add_nodes_from([(1, {'type': 'A'}), (2, {'type': 'B'}), (3, {'type': 'A'})])
        G.add_edges_from([(1, 2), (2, 3)])
        metapath = ['A', 'B', 'A']
        model = train_metapath2vec(G, metapath, walk_length=3, num_walks=10)

        # Test if the model has been trained
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
