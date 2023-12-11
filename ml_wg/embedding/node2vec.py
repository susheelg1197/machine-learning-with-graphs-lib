import random
import numpy as np
from gensim.models import Word2Vec
import networkx as nx

class Node2Vec:
    def __init__(self, graph, dimensions, walk_length, num_walks, p, q):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return hyperparameter
        self.q = q  # In-out hyperparameter

    def _get_alias_edges(self):
        # Helper function to create alias tables for edges
        pass

    def _biased_random_walk(self, start_node):
        # Generate a single random walk
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))
            if cur_neighbors:
                walk.append(random.choice(cur_neighbors))
            else:
                break
        return [str(node) for node in walk]

    def generate_walks(self):
        # Generate walks for each node
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.nodes():
                walks.append(self._biased_random_walk(node))
        return walks



    def train(self):
        walks = self.generate_walks()
        model = Word2Vec(walks, vector_size=self.dimensions, window=10, min_count=0, sg=1, workers=4)
        self.embeddings = {node: model.wv[str(node)] for node in self.graph.nodes()}

    def get_embedding(self, node):
        return self.embeddings.get(str(node), np.zeros(self.dimensions))

