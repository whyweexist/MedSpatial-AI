"""Stable scene-node feature extraction."""

import hashlib

import torch

from app.awm.schema import SceneNode


def node_features(nodes: list[SceneNode], dimension: int = 16) -> torch.Tensor:
    rows = []
    for node in nodes:
        digest = hashlib.sha256(f"{node.node_type}:{node.label}".encode()).digest()
        rows.append([(digest[index] / 127.5) - 1.0 for index in range(dimension)])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.empty(0, dimension)
