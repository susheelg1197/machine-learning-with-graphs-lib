import unittest
import torch
from ml_wg.gnn.gatv2 import GATv2

class TestGATv2(unittest.TestCase):
    def test_gatv2_forward(self):
        num_nodes = 10
        in_features = 5
        nhid = 8
        nout = 2
        nheads = 2

        model = GATv2(nfeat=in_features, nhid=nhid, nout=nout, nheads=nheads)
        x = torch.rand((num_nodes, in_features))
        adj = torch.randint(0, 2, (num_nodes, num_nodes))

        output = model(x, adj)
        self.assertEqual(output.shape, (num_nodes, nout))

if __name__ == '__main__':
    unittest.main()
