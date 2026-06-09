from app.realtime.base import RealtimeAdapter
from app.realtime.events import RealtimeEvent


class SpacetimeDBAdapter(RealtimeAdapter):
    """Explicit adapter placeholder; no network connection occurs until configured."""

    def __init__(self, client=None) -> None:
        self.client = client

    async def publish(self, event: RealtimeEvent) -> None:
        if self.client is None:
            raise RuntimeError("SpacetimeDB client is not configured")
        await self.client.publish(event.model_dump(mode="json"))

    async def history(self, topic: str, limit: int = 100) -> list[RealtimeEvent]:
        if self.client is None:
            raise RuntimeError("SpacetimeDB client is not configured")
        return [RealtimeEvent.model_validate(item) for item in await self.client.history(topic, limit)]
