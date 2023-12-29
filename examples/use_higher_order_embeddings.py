import networkx as nx
from ml_wg.embedding.higher_order_embeddings import higher_order_embeddings

# Example graph
G = nx.karate_club_graph()

# Generate embeddings
embeddings = higher_order_embeddings(G, motif_size=3)

# Print embeddings for a node
print(embeddings[0])  # Example node
