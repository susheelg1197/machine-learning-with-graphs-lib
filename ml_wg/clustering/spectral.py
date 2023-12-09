import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh

class SpectralClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, adj_matrix):
        # Compute the normalized Laplacian
        degree_matrix = np.diag(adj_matrix.sum(axis=1))
        laplacian = degree_matrix - adj_matrix
        norm_laplacian = np.dot(np.linalg.inv(degree_matrix), laplacian)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(norm_laplacian, k=self.n_clusters, which='SM')
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        return kmeans.fit_predict(eigenvectors)
