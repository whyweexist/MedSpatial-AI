"""Evidence-preserving graph queries."""

from app.awm.schema import SceneGraph


class GraphReasoner:
    def related(self, graph: SceneGraph, node_id: str) -> dict:
        neighbors = []
        evidence: set[str] = set()
        for edge in graph.edges:
            if node_id in {edge.source_id, edge.target_id}:
                other = edge.target_id if edge.source_id == node_id else edge.source_id
                neighbors.append({"node_id": other, "relation": edge.relation, "confidence": edge.confidence})
                evidence.update(edge.evidence_ids)
        return {"node_id": node_id, "neighbors": neighbors, "evidence_ids": sorted(evidence)}

    def is_isolated(self, graph: SceneGraph, node_id: str) -> bool:
        return not any(node_id in {edge.source_id, edge.target_id} for edge in graph.edges)
