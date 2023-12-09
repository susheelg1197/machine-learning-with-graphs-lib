import torch
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

class HierarchicalClustering:
    def __init__(self, method='average'):
        self.method = method

    def fit(self, adj_matrix):
        # Convert adjacency matrix to distance matrix
        distance_matrix = 1 - adj_matrix
        # Convert to condensed matrix format required by SciPy
        condensed_dist_matrix = squareform(distance_matrix)
        # Perform hierarchical clustering
        self.linkage_matrix = sch.linkage(condensed_dist_matrix, method=self.method)

    def get_clusters(self, num_clusters):
        return sch.fcluster(self.linkage_matrix, num_clusters, criterion='maxclust')
