import networkx as nx
from ml_wg.embedding.node2vec import Node2Vec

def test_node2vec():
    G = nx.karate_club_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=80, p=1, q=1)
    node2vec.train()
    embedding = node2vec.get_embedding(0)
    assert len(embedding) == 64, "Embedding size should be 64."


if __name__ == "__main__":
    test_node2vec()
