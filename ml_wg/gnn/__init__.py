# gnn/__init__.py

# Import the main classes/functions from your modules so they can be accessed directly from the gnn package:
from .gcn import GCN
from .gat import GAT
from .grn import GRN
from .gae import GAE
from .generative import GenerativeGNN
# ... any other imports as necessary ...

# You can also include any necessary initialization code for the gnn package here.

# Define what should be available for 'from gnn_package.gnn import *'
__all__ = ['GCN', 'GAT', 'GRN', 'GAE', 'GenerativeGNN']
