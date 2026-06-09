"""Build a compact, structured reasoning context."""

from app.awm.schema import AnatomicalWorldModel


def build_context(awm: AnatomicalWorldModel, selected_finding_id: str | None = None) -> dict:
    findings = awm.findings
    if selected_finding_id:
        findings = [item for item in findings if item.id == selected_finding_id]
    return {
        "study": awm.study.model_dump(mode="json"),
        "view": awm.current_view.model_dump(mode="json"),
        "findings": [item.model_dump(mode="json") for item in findings],
        "structures": [item.model_dump(mode="json") for item in awm.structures],
        "measurements": [item.model_dump(mode="json") for item in awm.measurements],
        "scene_graph": awm.scene_graph.model_dump(mode="json"),
        "latent_state": awm.latent_state.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in awm.evidence],
    }
