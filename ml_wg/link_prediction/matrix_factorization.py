import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

def matrix_factorization(G, num_factors=2, regularization=0.01, is_directed=False):
    # Get adjacency matrix (sparse format)
    A = nx.adjacency_matrix(G).asfptype()

    # Add regularization
    A_reg = A + regularization * csr_matrix(np.ones(A.shape))

    # Singular Value Decomposition
    U, s, Vt = svds(A_reg, k=num_factors)

    # Reconstruct the matrix (approximation)
    A_hat = U @ np.diag(s) @ Vt

    # For directed graphs, factorize A and A transpose separately
    if is_directed:
        At_reg = A.T + regularization * csr_matrix(np.ones(A.shape))
        Ut, st, Vtt = svds(At_reg, k=num_factors)
        A_hat_directed = Ut @ np.diag(st) @ Vtt
        A_hat = (A_hat + A_hat_directed) / 2  # Averaging the reconstructions

    return A_hat
