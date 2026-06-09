"""Evidence-grounded answer generation without diagnostic certainty."""

from pydantic import BaseModel, Field

from app.awm.schema import AnatomicalWorldModel, EvidenceAnchor


class GroundedAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence_anchors: list[EvidenceAnchor]
    provenance_ids: list[str]
    model_version: str
    source_modality: str
    grounding: str
    insufficient_evidence: bool = False


def answer_question(awm: AnatomicalWorldModel, question: str) -> GroundedAnswer:
    evidence_index = {item.id: item for item in awm.evidence}
    question_terms = set(question.lower().replace("?", "").split())
    ranked = sorted(
        awm.findings,
        key=lambda item: (
            len(question_terms & set((item.label + " " + item.description).lower().split())),
            item.confidence,
        ),
        reverse=True,
    )
    if not ranked:
        return GroundedAnswer(
            answer="The available world model does not contain enough evidence to answer this question.",
            confidence=0.0, uncertainty=1.0, evidence_anchors=[],
            provenance_ids=[], model_version="AWM-JGEM-local-1",
            source_modality=awm.study.modality.value,
            grounding="insufficient_evidence", insufficient_evidence=True,
        )
    finding = ranked[0]
    anchors = [evidence_index[item] for item in finding.evidence_ids if item in evidence_index]
    if not anchors:
        return GroundedAnswer(
            answer="A candidate finding exists, but its supporting evidence is unavailable, so no clinical claim can be made.",
            confidence=0.0, uncertainty=1.0, evidence_anchors=[],
            provenance_ids=[], model_version=finding.model.version,
            source_modality=awm.study.modality.value,
            grounding=finding.grounding.value, insufficient_evidence=True,
        )
    answer = (
        f"The model identified a candidate {finding.label.lower()} with "
        f"{finding.confidence:.0%} confidence and {finding.uncertainty:.0%} uncertainty. "
        f"This is not a confirmed diagnosis. {finding.description}"
    )
    return GroundedAnswer(
        answer=answer, confidence=finding.confidence, uncertainty=finding.uncertainty,
        evidence_anchors=anchors,
        provenance_ids=[record.id for record in awm.provenance if finding.id in record.output_ids],
        model_version=finding.model.version, source_modality=awm.study.modality.value,
        grounding=finding.grounding.value,
    )
