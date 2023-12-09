import networkx as nx
from ml_wg.embedding.deepwalk import DeepWalk

def main():
    G = nx.karate_club_graph()  # Example graph
    deepwalk = DeepWalk(G, walk_length=10, num_walks=80, dimensions=64)
    deepwalk.train()

    # Example: Get embedding for a node
    node_id = 0
    embedding = deepwalk.get_embedding(node_id)
    print(f"Embedding for node {node_id}: {embedding}")

if __name__ == "__main__":
    main()
