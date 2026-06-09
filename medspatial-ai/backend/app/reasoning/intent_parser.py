"""Convert user requests into non-executable structured intents."""

from app.policy.intent_schema import Action, Intent


def parse_question(
    question: str, actor_id: str, actor_role: str, study_id: str,
    deidentified_study_id: str, modality: str,
) -> Intent:
    return Intent(
        actor_id=actor_id,
        actor_role=actor_role,
        deidentified_study_id=deidentified_study_id,
        study_id=study_id,
        modality=modality,
        requested_action=Action.ASK_CLINICAL_QUESTION,
        parameters={"question": question},
        provenance_requirement=True,
        risk_level="medium",
    )
