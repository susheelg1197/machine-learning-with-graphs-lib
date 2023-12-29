import unittest
import networkx as nx
from ml_wg.classification.wl_kernel import weisfeiler_lehman_kernel

class TestWeisfeilerLehmanKernel(unittest.TestCase):

    def test_weisfeiler_lehman_kernel(self):
        G1 = nx.complete_graph(5)
        G2 = nx.path_graph(5)
    
        kernel_value = weisfeiler_lehman_kernel(G1, G2, h=3)

        self.assertIsInstance(kernel_value, (int, float))

if __name__ == '__main__':
    unittest.main()
