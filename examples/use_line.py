import networkx as nx
from ml_wg.embedding.line import LINE

def main():
    # Create or load a graph
    G = nx.karate_club_graph()

    # Initialize and train the LINE model
    line_model = LINE(G, dimensions=64)
    line_model.train()

    # Retrieve and print embeddings for some nodes
    for node_id in range(5):  # Example: Nodes 0 to 4
        embedding = line_model.get_embedding(node_id)
        print(f"Embedding for node {node_id}: {embedding}")

if __name__ == "__main__":
    main()
