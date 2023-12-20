import networkx as nx
from collections import Counter
from itertools import combinations

def graphlet_degree_vector(G, k=3):
    def find_graphlets(node, k):
        return {frozenset(c) for c in combinations(G.neighbors(node), k-1)}

    graphlets = Counter()
    for node in G.nodes():
        graphlets.update(find_graphlets(node, k))

    return dict(graphlets)

def graphlet_kernel(G1, G2, k=3):
    gdv1 = graphlet_degree_vector(G1, k)
    gdv2 = graphlet_degree_vector(G2, k)

    kernel_value = sum(gdv1.get(g, 0) * gdv2.get(g, 0) for g in set(gdv1) | set(gdv2))
    return kernel_value
