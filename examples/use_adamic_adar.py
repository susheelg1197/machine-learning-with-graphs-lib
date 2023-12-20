import networkx as nx
from ml_wg.link_prediction.adamic_adar import adamic_adar_index

G = nx.Graph([(1, 2), (2, 3), (3, 4)])
print(adamic_adar_index(G, 1, 3))
