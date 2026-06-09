"""Pure PyTorch graph attention fallback."""

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention = nn.Linear(output_dim * 2, 1, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features)
        count = hidden.shape[0]
        left = hidden.unsqueeze(1).expand(count, count, -1)
        right = hidden.unsqueeze(0).expand(count, count, -1)
        scores = F.leaky_relu(self.attention(torch.cat([left, right], dim=-1)).squeeze(-1))
        scores = scores.masked_fill(~adjacency.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return F.gelu(weights @ hidden)


class AnatomicalGNN(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layer1 = GraphAttentionLayer(input_dim, hidden_dim)
        self.layer2 = GraphAttentionLayer(hidden_dim, hidden_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(features, adjacency), adjacency)
