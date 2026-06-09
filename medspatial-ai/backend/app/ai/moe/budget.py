"""Inference budget contracts."""

from pydantic import BaseModel, Field


class InferenceBudget(BaseModel):
    device: str = "cpu"
    available_memory_mb: int = Field(default=1024, ge=64)
    maximum_latency_ms: int = Field(default=30_000, ge=1)
    allow_remote: bool = False
    allow_gpu: bool = True
