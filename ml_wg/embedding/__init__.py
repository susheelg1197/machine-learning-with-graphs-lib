# Import the main classes/functions from your modules so they can be accessed directly from the clustering package:
# from .line import HierarchicalClustering
from .deepwalk import DeepWalk
from .line import LINE
from .node2vec import Node2Vec
# from .node2vec import 
# ... any other clustering methods you implement ...

# You can also include any necessary initialization code for the clustering package here.

# Define what should be available for 'from clustering_package.clustering import *'
__all__ = ['DeepWalk','LINE', 'Node2Vec']
