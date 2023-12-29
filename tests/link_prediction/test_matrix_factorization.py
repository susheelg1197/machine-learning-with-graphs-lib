import unittest
import networkx as nx
from ml_wg.link_prediction.matrix_factorization import matrix_factorization

class TestMatrixFactorization(unittest.TestCase):
    def test_matrix_factorization(self):
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        A_hat = matrix_factorization(G, num_factors=2)
        self.assertEqual(A_hat.shape, (4, 4))

if __name__ == '__main__':
    unittest.main()
