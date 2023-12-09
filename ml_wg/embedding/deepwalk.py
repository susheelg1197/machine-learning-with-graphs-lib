import random
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, dimensions):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions

    def generate_walks(self):
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.nodes():
                walk = [str(node)]  # Convert node ID to string
                for _ in range(self.walk_length - 1):
                    neighbors = [n for n in self.graph.neighbors(node)]
                    if neighbors:
                        next_node = random.choice(neighbors)
                        walk.append(str(next_node))  # Convert neighbor node ID to string
                    else:
                        break
                walks.append(walk)
        return walks


    def train(self):
        walks = self.generate_walks()
        model = Word2Vec(walks, vector_size=self.dimensions, window=5, min_count=0, sg=1, workers=2)
        self.embeddings = {node: model.wv[str(node)] for node in self.graph.nodes()}


    def get_embedding(self, node):
        return self.embeddings.get(node, None)
