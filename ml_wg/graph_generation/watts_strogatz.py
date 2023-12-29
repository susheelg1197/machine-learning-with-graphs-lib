import networkx as nx
import numpy as np

def generate_watts_strogatz(n, k, p, seed=None, add_weights=False, add_attributes=False):
    """
    Generate a Watts-Strogatz small-world graph with additional features.

    Parameters:
    - n (int): Number of nodes.
    - k (int): Each node is connected to k nearest neighbors in ring topology.
    - p (float): The probability of rewiring each edge.
    - seed (int, optional): Random seed.
    - add_weights (bool): Whether to add random weights to edges.
    - add_attributes (bool): Whether to add custom attributes to nodes.

    Returns:
    - networkx.Graph: A Watts-Strogatz graph.
    """
    G = nx.watts_strogatz_graph(n, k, p, seed)

    if add_weights:
        for u, v in G.edges():
            G.edges[u, v]['weight'] = np.random.rand()

    if add_attributes:
        for node in G.nodes():
            G.nodes[node]['attribute'] = np.random.rand()

    return G
