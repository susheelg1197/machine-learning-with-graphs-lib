import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1):
        super(GraphTransformerLayer, self).__init__()
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads)

        # Feed-forward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, in_features)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class GraphTransformer(nn.Module):
    def __init__(self, n_layers, in_features, out_features, num_heads=1):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([GraphTransformerLayer(in_features, out_features, num_heads) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
