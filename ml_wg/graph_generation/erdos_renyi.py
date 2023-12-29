import networkx as nx
import numpy as np
from scipy.stats import truncnorm

def generate_erdos_renyi(n, p, seed=None, directed=False, connected=False, weights=False, communities=1, intra_p=None, inter_p=None):
    """
    Generate an Erdos-Renyi graph with additional features like communities, custom weight distributions, and more.
    
    Parameters:
    - n: Number of nodes
    - p: Probability of edge creation
    - seed: Seed for random number generator
    - directed: Boolean flag to create directed graph
    - connected: Ensure the graph is connected
    - weights: Assign weights to edges
    - communities: Number of communities within the graph
    - intra_p: Probability of edge creation within communities
    - inter_p: Probability of edge creation between communities
    
    Returns:
    - G: An Erdos-Renyi graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    if communities > 1 and (intra_p is not None or inter_p is not None):
        # Generate a stochastic block model for community structure
        sizes = [n // communities] * communities
        probs = np.full((communities, communities), inter_p if inter_p is not None else p)
        np.fill_diagonal(probs, intra_p if intra_p is not None else p)
        G = nx.stochastic_block_model(sizes, probs, seed=seed, directed=directed)
    else:
        G = nx.gnp_random_graph(n, p, seed, directed)
    
    if connected and not directed and communities == 1:
        # Adding edges to ensure connectivity, if disconnected
        while not nx.is_connected(G):
            for i in range(n):
                if not nx.has_path(G, 0, i):
                    G.add_edge(0, i)
                    break

    if weights:
        for u, v in G.edges():
            # Assign weights using a truncated normal distribution
            G.edges[u, v]['weight'] = truncnorm(a=0, b=np.inf, loc=0.5, scale=0.1).rvs()

    return G
