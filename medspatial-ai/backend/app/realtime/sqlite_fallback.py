import asyncio
import sqlite3
from pathlib import Path

from app.realtime.base import RealtimeAdapter
from app.realtime.events import RealtimeEvent


class SQLiteRealtimeAdapter(RealtimeAdapter):
    def __init__(self, path: str = "./data/realtime.sqlite") -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(id TEXT PRIMARY KEY, topic TEXT, study_id TEXT, timestamp TEXT, payload TEXT)"
        )
        return connection

    async def publish(self, event: RealtimeEvent) -> None:
        def write() -> None:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                    (event.id, event.topic, event.study_id, event.timestamp.isoformat(), event.model_dump_json()),
                )
        await asyncio.to_thread(write)

    async def history(self, topic: str, limit: int = 100) -> list[RealtimeEvent]:
        def read() -> list[str]:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT payload FROM events WHERE topic = ? ORDER BY timestamp DESC LIMIT ?",
                    (topic, limit),
                ).fetchall()
            return [row[0] for row in rows]
        return [RealtimeEvent.model_validate_json(item) for item in await asyncio.to_thread(read)]
