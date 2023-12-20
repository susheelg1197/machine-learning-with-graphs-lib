# gnn/__init__.py

# Import the main classes/functions from your modules so they can be accessed directly from the gnn package:
from .gcn import GCN
from .gat import GAT
from .gatv2 import GATv2  # Import for GATv2
from .gae import GraphAutoencoder
from .generative import GraphGenerativeNN
from .graphsage import GraphSAGE  # Import for GraphSAGE
from .graph_transformer import GraphTransformer  # Import for Graph Transformer
from .tgn import TGN  # Import for Temporal Graph Network
from .unet import GraphUNet  # Import for Graph U-Net
from .hgn import HGN  # Import for Hamiltonian Graph Network

# You can also include any necessary initialization code for the gnn package here.

# Define what should be available for 'from gnn_package.gnn import *'
__all__ = [
    'GCN', 
    'GAT', 
    'GATv2',  # Include GATv2 in the export list
    'GraphAutoencoder', 
    'GraphGenerativeNN',
    'GraphSAGE',  # Include GraphSAGE in the export list
    'GraphTransformer',  # Include Graph Transformer in the export list
    'TGN',  # Include Temporal Graph Network in the export list
    'GraphUNet',  # Include Graph U-Net in the export list
    'HGN'  # Include Hamiltonian Graph Network in the export list
]
