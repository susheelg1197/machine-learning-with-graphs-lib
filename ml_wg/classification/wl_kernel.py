import networkx as nx
from grakel.kernels import WeisfeilerLehman
from grakel import Graph

def convert_networkx_to_grakel(G):
    # Convert a NetworkX graph to a format compatible with GraKeL
    nodes = list(G.nodes(data=True))
    node_labels = {i: data.get("label", 0) for i, data in nodes}
    edges = [(u, v, G[u][v].get("weight", 1.0)) for u, v in G.edges()]
    return Graph(edges, node_labels=node_labels)

def weisfeiler_lehman_kernel(G1, G2, h=3):
    # Convert NetworkX graphs to Grakel format
    G1_grakel = convert_networkx_to_grakel(G1)
    G2_grakel = convert_networkx_to_grakel(G2)

    # Initialize the Weisfeiler-Lehman kernel
    wl_kernel = WeisfeilerLehman(n_iter=h, normalize=True)

    # Compute and return the kernel
    return wl_kernel.fit_transform([G1_grakel, G2_grakel])[0, 1]
