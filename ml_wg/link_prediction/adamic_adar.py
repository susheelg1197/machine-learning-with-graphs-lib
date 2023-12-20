import networkx as nx

def adamic_adar_index(G, u, v, is_directed=False, normalize=False):
    if not G.has_node(u) or not G.has_node(v):
        return 0

    # Choose neighbor function based on graph type
    neighbors_fn = G.neighbors
    if is_directed:
        neighbors_fn = lambda n: set(G.predecessors(n)).union(G.successors(n))

    common_neighbors = set(neighbors_fn(u)).intersection(neighbors_fn(v))
    
    # Calculate Adamic-Adar index
    aa_index = sum(1 / nx.degree(G, w) for w in common_neighbors)

    # Normalization (if required)
    if normalize:
        max_index = sum(1 / nx.degree(G, w) for w in neighbors_fn(u))
        aa_index = aa_index / max_index if max_index > 0 else 0

    return aa_index
