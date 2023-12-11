import networkx as nx
from ml_wg.embedding.node2vec import Node2Vec

def main():
    G = nx.karate_club_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=80, p=1, q=1)
    node2vec.train()

    for node_id in range(5):
        embedding = node2vec.get_embedding(node_id)
        print(f"Embedding for node {node_id}: {embedding}")

if __name__ == "__main__":
    main()
