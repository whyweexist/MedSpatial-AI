"""Append-only, PHI-aware JSONL audit logging."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.policy.intent_schema import Intent
from app.policy.policy_engine import PolicyDecision


class AuditLogger:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.AUDIT_DIR)
        self._lock = asyncio.Lock()

    async def log(self, intent: Intent, decision: PolicyDecision) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "policy_decision",
            "intent_id": intent.intent_id,
            "trace_id": intent.trace_id,
            "actor_id": intent.actor_id,
            "actor_role": intent.actor_role,
            "study_id": intent.study_id,
            "action": intent.requested_action.value,
            "decision": decision.decision,
            "reason_code": decision.reason_code,
        }
        async with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"{intent.study_id}.jsonl"
            await asyncio.to_thread(self._append, path, event)

    @staticmethod
    def _append(path: Path, event: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    async def read(self, study_id: str) -> list[dict]:
        path = self.root / f"{study_id}.jsonl"
        if not path.exists():
            return []
        lines = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return [json.loads(line) for line in lines.splitlines() if line]
