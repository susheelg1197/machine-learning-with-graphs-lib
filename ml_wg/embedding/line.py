import numpy as np
import random
from gensim.models import Word2Vec
import networkx as nx

class LINE:
    def __init__(self, graph, dimensions):
        self.graph = graph
        self.dimensions = dimensions
        self.embeddings = {}

    def generate_edges(self):
        edges = []
        for edge in self.graph.edges():
            edges.append([str(edge[0]), str(edge[1])])
        return edges

    def train(self):
        edges = self.generate_edges()
        model = Word2Vec(edges, vector_size=self.dimensions, window=1, min_count=0, sg=1, workers=4)
        # Ensure embeddings for all nodes
        for node in self.graph.nodes():
            self.embeddings[node] = model.wv[str(node)] if str(node) in model.wv else np.zeros(self.dimensions)

    def get_embedding(self, node):
        return self.embeddings.get(node, None)
