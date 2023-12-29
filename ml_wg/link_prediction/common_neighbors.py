import networkx as nx

def common_neighbors(G, u, v, directed=False, weighted=False, return_list=False, neighbor_selector=None):
    if not G.has_node(u) or not G.has_node(v):
        return 0 if not return_list else []

    # Define neighbor selector function
    if directed:
        if neighbor_selector is not None:
            neighbors_fn = neighbor_selector
        else:
            neighbors_fn = lambda n: set(G.predecessors(n)).union(G.successors(n))
    else:
        neighbors_fn = G.neighbors

    common_neigh = set(neighbors_fn(u)).intersection(neighbors_fn(v))

    if return_list:
        return list(common_neigh)

    if weighted:
        return sum(G[u][w].get('weight', 1) + G[v][w].get('weight', 1) for w in common_neigh)
    else:
        return len(common_neigh)
