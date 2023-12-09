import networkx as nx
from ml_wg.embedding.deepwalk import DeepWalk

def test_deepwalk():
    G = nx.karate_club_graph()
    deepwalk = DeepWalk(G, walk_length=10, num_walks=5, dimensions=64)
    deepwalk.train()
    embedding = deepwalk.get_embedding(0)
    assert len(embedding) == 64, "Embedding size should be 64."

if __name__ == "__main__":
    test_deepwalk()
    print("DeepWalk test passed.")
