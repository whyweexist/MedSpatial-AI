"""Build scene graphs from the AWM."""

from app.awm.schema import AnatomicalWorldModel, SceneEdge, SceneGraph, SceneNode


def build_scene_graph(awm: AnatomicalWorldModel) -> SceneGraph:
    nodes = [
        SceneNode(
            id=structure.id,
            node_type="structure",
            label=structure.name,
            evidence_ids=structure.evidence_ids,
            attributes={
                "region": structure.body_region.value,
                "system": structure.system.value,
                "grounding": structure.grounding.value,
                "confidence": structure.confidence,
                "uncertainty": structure.uncertainty,
            },
        )
        for structure in awm.structures
    ]
    nodes.extend(
        SceneNode(
            id=finding.id,
            node_type="finding",
            label=finding.label,
            evidence_ids=finding.evidence_ids,
            attributes={"confidence": finding.confidence, "uncertainty": finding.uncertainty},
        )
        for finding in awm.findings
    )
    edges: list[SceneEdge] = []
    structure_ids = {structure.id for structure in awm.structures}
    for structure in awm.structures:
        if structure.parent_id and structure.parent_id in structure_ids:
            edges.append(SceneEdge(source_id=structure.parent_id, target_id=structure.id, relation="contains"))
    for finding in awm.findings:
        for structure_id in finding.structure_ids:
            if structure_id in structure_ids:
                edges.append(SceneEdge(
                    source_id=finding.id,
                    target_id=structure_id,
                    relation="suspicious_for",
                    confidence=finding.confidence,
                    evidence_ids=finding.evidence_ids,
                ))
    return SceneGraph(nodes=nodes, edges=edges)
