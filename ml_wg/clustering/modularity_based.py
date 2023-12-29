import numpy as np
import networkx as nx
import community as community_louvain

class ModularityBasedClustering:
    def __init__(self):
        self.partition = None

    def fit(self, adj_matrix):
        G = nx.Graph()
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] != 0:
                    G.add_edge(i, j, weight=adj_matrix[i, j])
        
        self.partition = community_louvain.best_partition(G)

    def get_clusters(self):
        return self.partition
