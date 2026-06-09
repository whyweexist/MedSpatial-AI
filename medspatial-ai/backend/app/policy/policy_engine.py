"""Deterministic, deny-by-default policy evaluation."""

from pydantic import BaseModel

from app.policy.intent_schema import Action, Intent


class PolicyDecision(BaseModel):
    allowed: bool
    decision: str
    reason_code: str
    user_explanation: str
    required_controls: list[str] = []
    trace_id: str


class PolicyEngine:
    CLINICAL_ROLES = {"radiologist", "clinician", "admin"}
    ADMIN_ACTIONS = {Action.DELETE_STUDY, Action.EXPORT_RAW_STUDY}
    NETWORK_ACTIONS = {Action.EXTERNAL_DATA_TRANSFER, Action.SHARE_SESSION}

    def evaluate(self, intent: Intent) -> PolicyDecision:
        if intent.requested_action in self.ADMIN_ACTIONS and intent.actor_role != "admin":
            return self._deny(intent, "ROLE_REQUIRED", "This action requires an administrator role.")
        if intent.requested_action in self.NETWORK_ACTIONS:
            if not intent.network_destination:
                return self._deny(intent, "DESTINATION_REQUIRED", "A network destination must be declared.")
            if not intent.user_confirmed:
                return self._deny(intent, "CONFIRMATION_REQUIRED", "Explicit user confirmation is required.")
        if intent.requested_action == Action.EXPORT_RAW_STUDY:
            if intent.sensitivity_level != "phi":
                return self._deny(intent, "CLASSIFICATION_MISMATCH", "Raw export requires PHI classification.")
            if not intent.user_confirmed:
                return self._deny(intent, "CONFIRMATION_REQUIRED", "Raw study export requires confirmation.")
        if intent.requested_action == Action.ASK_CLINICAL_QUESTION and not intent.provenance_requirement:
            return self._deny(intent, "EVIDENCE_REQUIRED", "Clinical answers must preserve evidence provenance.")
        if intent.requested_action in {
            Action.SEGMENT_ANATOMY, Action.MEASURE_LESION, Action.ASK_CLINICAL_QUESTION
        } and intent.actor_role not in self.CLINICAL_ROLES:
            return self._deny(intent, "CLINICAL_ROLE_REQUIRED", "A clinical role is required for this action.")
        return PolicyDecision(
            allowed=True,
            decision="allow",
            reason_code="POLICY_SATISFIED",
            user_explanation="The requested action satisfies the current local policy.",
            required_controls=["audit_log", "provenance"] if intent.provenance_requirement else ["audit_log"],
            trace_id=intent.trace_id,
        )

    @staticmethod
    def _deny(intent: Intent, code: str, explanation: str) -> PolicyDecision:
        return PolicyDecision(
            allowed=False, decision="deny", reason_code=code,
            user_explanation=explanation, required_controls=[],
            trace_id=intent.trace_id,
        )
