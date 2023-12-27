import networkx as nx
import numpy as np
from ml_wg.graph_signal_processing.spectral_filtering import spectral_filtering

def main():
    # Generate a graph
    G = nx.karate_club_graph()

    # Create a synthetic signal on the graph (e.g., node feature or measurement)
    signal = np.random.rand(G.number_of_nodes())

    # Apply spectral filtering
    lowpass_filtered = spectral_filtering(G, signal, filter_type='lowpass', cutoff=0.5)
    highpass_filtered = spectral_filtering(G, signal, filter_type='highpass', cutoff=0.5)
    bandpass_filtered = spectral_filtering(G, signal, filter_type='bandpass', cutoff=(0.2, 0.8))

    # Display the results
    print("Original Signal:", signal)
    print("Lowpass Filtered Signal:", lowpass_filtered)
    print("Highpass Filtered Signal:", highpass_filtered)
    print("Bandpass Filtered Signal:", bandpass_filtered)

if __name__ == '__main__':
    main()
