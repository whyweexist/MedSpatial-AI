"""The only permitted bridge from reasoning to privileged execution."""

from collections.abc import Awaitable, Callable
from typing import Any

from app.policy.audit_logger import AuditLogger
from app.policy.intent_schema import Intent
from app.policy.policy_engine import PolicyDecision, PolicyEngine


class ExecutorGateway:
    def __init__(self) -> None:
        self.policy = PolicyEngine()
        self.audit = AuditLogger()

    async def evaluate(self, intent: Intent) -> PolicyDecision:
        decision = self.policy.evaluate(intent)
        await self.audit.log(intent, decision)
        return decision

    async def execute(
        self, intent: Intent, handler: Callable[[Intent], Awaitable[Any]]
    ) -> Any:
        decision = await self.evaluate(intent)
        if not decision.allowed:
            return {"executed": False, "policy": decision.model_dump()}
        result = await handler(intent)
        return {"executed": True, "policy": decision.model_dump(), "result": result}
