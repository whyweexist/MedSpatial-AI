"""Human-readable graph explanation."""

from app.awm.schema import SceneGraph


def explain_neighbors(graph: SceneGraph, node_id: str) -> list[str]:
    names = {node.id: node.label for node in graph.nodes}
    return [
        f"{names.get(edge.source_id, edge.source_id)} {edge.relation} {names.get(edge.target_id, edge.target_id)}"
        for edge in graph.edges if node_id in {edge.source_id, edge.target_id}
    ]
