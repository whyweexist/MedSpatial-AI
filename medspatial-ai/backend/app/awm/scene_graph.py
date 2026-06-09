"""Scene graph validation helpers."""

from app.awm.schema import SceneGraph


def validate_scene_graph(graph: SceneGraph) -> SceneGraph:
    node_ids = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise ValueError(f"Scene edge {edge.id} references an unknown node")
    return graph
