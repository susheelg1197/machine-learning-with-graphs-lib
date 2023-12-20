import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGraphNetwork(nn.Module):
    def __init__(self, node_features, edge_features, temporal_features):
        super(TemporalGraphNetwork, self).__init__()
        # Example: Layers for handling node, edge, and temporal features
        self.node_encoder = nn.Linear(node_features, 64)
        self.edge_encoder = nn.Linear(edge_features, 64)
        self.temporal_encoder = nn.Linear(temporal_features, 64)

        # Example: Additional layers for the TGN model
        self.combined_layer = nn.Linear(192, 128)  # Combining node, edge, temporal features

    def forward(self, node_embeddings, edge_embeddings, temporal_embeddings):
        # Encoding node, edge, and temporal features
        encoded_nodes = self.node_encoder(node_embeddings)
        encoded_edges = self.edge_encoder(edge_embeddings)
        encoded_temporal = self.temporal_encoder(temporal_embeddings)

        # Combining the encoded features
        combined = torch.cat([encoded_nodes, encoded_edges, encoded_temporal], dim=1)
        updated_embeddings = self.combined_layer(combined)
        return updated_embeddings
