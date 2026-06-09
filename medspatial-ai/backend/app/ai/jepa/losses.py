"""Composable JEPA training losses."""

import torch
from torch.nn import functional as F


def masked_latent_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1).mean()


def cross_slice_consistency(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens[:, 1:] - tokens[:, :-1]).pow(2).mean() if tokens.shape[1] > 1 else tokens.sum() * 0


def adjacency_consistency(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(left, right, dim=-1).mean()


def topology_consistency(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.diff(dim=-1).abs().mean()
