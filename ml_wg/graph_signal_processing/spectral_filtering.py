import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs

def spectral_filtering(G, signal, filter_type='lowpass', cutoff=0.5, n_eigvals=None):
    """
    Apply spectral filtering to signals on a graph.

    Parameters:
    - G: NetworkX graph
    - signal: numpy array, signal on the graph
    - filter_type: str, type of filter ('lowpass', 'highpass', 'bandpass')
    - cutoff: float or tuple, cutoff frequency/frequencies for the filter
    - n_eigvals: int, number of eigenvalues to compute (None for all)

    Returns:
    - filtered_signal: numpy array, filtered signal
    """
    # Handle the case of disconnected graph
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Compute Laplacian matrix
    L = nx.normalized_laplacian_matrix(G).asfptype()

    # Use sparse eigenvalue solver for large graphs
    if n_eigvals is None or n_eigvals > L.shape[0]:
        eigvals, eigvecs = np.linalg.eigh(L.toarray())
    else:
        eigvals, eigvecs = eigs(L, k=n_eigvals, which='SM')
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    # Apply the spectral filter
    filtered_signal = np.zeros_like(signal)
    if filter_type == 'bandpass' and isinstance(cutoff, tuple):
        low, high = cutoff
    for i in range(len(eigvals)):
        include_eigval = False
        if filter_type == 'lowpass' and eigvals[i] < cutoff:
            include_eigval = True
        elif filter_type == 'highpass' and eigvals[i] > cutoff:
            include_eigval = True
        elif filter_type == 'bandpass' and low < eigvals[i] < high:
            include_eigval = True

        if include_eigval:
            filtered_signal += eigvecs[:, i] * (eigvecs[:, i].T @ signal)

    return filtered_signal
