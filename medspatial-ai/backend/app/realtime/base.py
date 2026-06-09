from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.realtime.events import RealtimeEvent


class RealtimeAdapter(ABC):
    @abstractmethod
    async def publish(self, event: RealtimeEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    async def history(self, topic: str, limit: int = 100) -> list[RealtimeEvent]:
        raise NotImplementedError

    async def subscribe(self, topic: str) -> AsyncIterator[RealtimeEvent]:
        raise NotImplementedError("Subscriptions are provider-specific")
