import unittest
import networkx as nx
from ml_wg.classification.graph_kernels import graphlet_kernel

class TestGraphletKernel(unittest.TestCase):

    def test_graphlet_kernel(self):
        G1 = nx.complete_graph(5)
        G2 = nx.path_graph(5)

        kernel_value = graphlet_kernel(G1, G2, k=3)

        self.assertIsInstance(kernel_value, (int, float))

if __name__ == '__main__':
    unittest.main()
