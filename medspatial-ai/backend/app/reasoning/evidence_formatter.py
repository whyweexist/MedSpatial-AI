"""Format evidence anchors for API/UI consumption."""

from app.awm.schema import EvidenceAnchor


def format_evidence(anchors: list[EvidenceAnchor]) -> list[dict]:
    return [anchor.model_dump(mode="json") for anchor in anchors]
