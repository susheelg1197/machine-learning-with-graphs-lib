import unittest
import networkx as nx
import numpy as np
from ml_wg.graph_signal_processing.spectral_filtering import spectral_filtering

class TestSpectralFiltering(unittest.TestCase):
    def setUp(self):
        # Create a simple graph for testing
        self.G = nx.cycle_graph(10)
        self.signal = np.random.rand(10)

    def test_lowpass_filtering(self):
        filtered_signal = spectral_filtering(self.G, self.signal, filter_type='lowpass', cutoff=0.5)
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_highpass_filtering(self):
        filtered_signal = spectral_filtering(self.G, self.signal, filter_type='highpass', cutoff=0.5)
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_bandpass_filtering(self):
        filtered_signal = spectral_filtering(self.G, self.signal, filter_type='bandpass', cutoff=(0.2, 0.8))
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_disconnected_graph(self):
        # Create a disconnected graph
        G_disconnected = nx.disjoint_union(nx.path_graph(5), nx.cycle_graph(5))

        # Create a signal vector for the largest connected component
        largest_cc = max(nx.connected_components(G_disconnected), key=len)
        signal = np.random.rand(len(largest_cc))

        # Perform spectral filtering on the largest connected component
        filtered_signal = spectral_filtering(G_disconnected, signal, cutoff=0.5)

        # Assertions (modify as needed)
        self.assertEqual(filtered_signal.shape[0], len(largest_cc))
        self.assertFalse(np.allclose(filtered_signal, np.zeros_like(signal)))


if __name__ == '__main__':
    unittest.main()
