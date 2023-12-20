import random
import gensim
import networkx as nx

def metapath2vec_random_walk(G, metapath, walk_length):
    walk = [str(random.choice([n for n in G.nodes if G.nodes[n]['type'] == metapath[0]]))]
    for _ in range(walk_length - 1):
        current = walk[-1]
        next_type = metapath[len(walk) % len(metapath)]
        neighbors = [str(n) for n in G.neighbors(int(current)) if G.nodes[n]['type'] == next_type]
        if not neighbors:  # No suitable neighbors
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk

def train_metapath2vec(G, metapath, walk_length, num_walks):
    walks = []
    for _ in range(num_walks):
        walks.extend(metapath2vec_random_walk(G, metapath, walk_length))

    model = gensim.models.Word2Vec(walks, vector_size=128, window=5, min_count=0, sg=1)
    return model
