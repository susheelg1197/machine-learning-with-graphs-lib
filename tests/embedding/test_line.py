import networkx as nx
from ml_wg.embedding.line import LINE

def test_line_embeddings():
    # Create a small test graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Initialize LINE model
    line_model = LINE(G, dimensions=64)
    line_model.train()

    # Test embeddings are generated for each node
    for node in G.nodes():
        embedding = line_model.get_embedding(node)
        assert embedding is not None, f"Embedding for node {node} should not be None"
        assert len(embedding) == 64, f"Embedding size for node {node} should be 64"

if __name__ == "__main__":
    test_line_embeddings()
    print("All tests passed.")
