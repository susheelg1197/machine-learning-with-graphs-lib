# Import the main classes/functions from your modules so they can be accessed directly from the clustering package:
from .hierarchical import HierarchicalClustering
from .spectral import SpectralClustering
# ... any other clustering methods you implement ...

# You can also include any necessary initialization code for the clustering package here.

# Define what should be available for 'from clustering_package.clustering import *'
__all__ = ['HierarchicalClustering', 'SpectralClustering']
