"""Convert scene edges to a self-connected adjacency matrix."""

import torch

from app.awm.schema import SceneGraph


def adjacency_matrix(graph: SceneGraph) -> torch.Tensor:
    index = {node.id: position for position, node in enumerate(graph.nodes)}
    matrix = torch.eye(len(index), dtype=torch.bool)
    for edge in graph.edges:
        if edge.source_id in index and edge.target_id in index:
            matrix[index[edge.source_id], index[edge.target_id]] = True
            matrix[index[edge.target_id], index[edge.source_id]] = True
    return matrix
