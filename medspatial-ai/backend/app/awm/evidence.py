"""Evidence helpers that prevent unsupported clinical output."""

from app.awm.schema import EvidenceAnchor, FindingCandidate


def require_evidence(finding: FindingCandidate) -> FindingCandidate:
    if not finding.evidence_ids:
        raise ValueError("Clinical finding candidates require at least one evidence anchor")
    return finding


def index_evidence(items: list[EvidenceAnchor]) -> dict[str, EvidenceAnchor]:
    return {item.id: item for item in items}
