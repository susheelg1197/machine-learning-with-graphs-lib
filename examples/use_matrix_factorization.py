import networkx as nx
from  ml_wg.link_prediction.matrix_factorization import matrix_factorization

G = nx.Graph([(1, 2), (2, 3), (3, 4)])
A_hat = matrix_factorization(G, num_factors=2)
print(A_hat)
