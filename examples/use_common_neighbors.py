import networkx as nx
from ml_wg.link_prediction.common_neighbors import common_neighbors

G = nx.Graph([(1, 2), (2, 3), (1, 3)])
print(common_neighbors(G, 1, 3))
