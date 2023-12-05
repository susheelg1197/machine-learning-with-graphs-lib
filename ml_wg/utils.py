# ml_wg/utils.py

import torch

def normalize_adjacency(adjacency_matrix):
    """
    Normalizes the adjacency matrix using the degree matrix.
    """
    adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(0))
    rowsum = adjacency_matrix.sum(1)
    degree_mat_inv_sqrt = torch.diag(rowsum.pow(-0.5))
    adjacency_matrix = degree_mat_inv_sqrt @ adjacency_matrix @ degree_mat_inv_sqrt
    return adjacency_matrix
