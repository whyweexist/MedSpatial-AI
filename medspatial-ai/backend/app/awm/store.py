"""Atomic, local-first JSON persistence for Anatomical World Models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from app.awm.schema import AnatomicalWorldModel, utc_now
from app.config import settings


class AWMStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.AWM_DIR)
        self._locks: dict[str, asyncio.Lock] = {}

    def _path(self, study_id: str) -> Path:
        if not study_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in study_id):
            raise ValueError("Invalid study identifier")
        return self.root / f"{study_id}.json"

    async def get(self, study_id: str) -> AnatomicalWorldModel | None:
        path = self._path(study_id)
        if not path.exists():
            return None
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return AnatomicalWorldModel.model_validate_json(text)

    async def save(self, awm: AnatomicalWorldModel) -> AnatomicalWorldModel:
        lock = self._locks.setdefault(awm.study.study_id, asyncio.Lock())
        async with lock:
            self.root.mkdir(parents=True, exist_ok=True)
            awm.updated_at = utc_now()
            path = self._path(awm.study.study_id)
            temporary = path.with_suffix(".json.tmp")
            payload = awm.model_dump_json(indent=2)
            await asyncio.to_thread(temporary.write_text, payload, encoding="utf-8")
            await asyncio.to_thread(temporary.replace, path)
        return awm

    async def update(
        self,
        study_id: str,
        updater: Callable[[AnatomicalWorldModel], AnatomicalWorldModel],
    ) -> AnatomicalWorldModel:
        awm = await self.get(study_id)
        if awm is None:
            raise KeyError(f"AWM not found for study {study_id}")
        return await self.save(updater(awm))


_store: AWMStore | None = None


def get_awm_store() -> AWMStore:
    global _store
    if _store is None:
        _store = AWMStore()
    return _store
