import networkx as nx
from ml_wg.embedding.metapath2vec import train_metapath2vec

# Example graph
G = nx.Graph()
G.add_nodes_from([(1, {'type': 'A'}), (2, {'type': 'B'}), (3, {'type': 'A'})])
G.add_edges_from([(1, 2), (2, 3)])

# Define metapath
metapath = ['A', 'B', 'A']

# Train the model
model = train_metapath2vec(G, metapath, walk_length=3, num_walks=10)

# Example: Retrieve vector for a node
print(model.wv['1'])
