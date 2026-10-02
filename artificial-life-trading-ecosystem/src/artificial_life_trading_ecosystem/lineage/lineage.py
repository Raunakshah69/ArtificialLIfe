"""Lineage tracking interface for agent ancestry and descendants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LineageRecord:
    """Record describing an agent and its ancestry."""

    agent_id: str = ""
    parent_a: str | None = None
    parent_b: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    offspring_ids: list[str] = field(default_factory=list)
    generation: int = 0
    reproduction_method: str = "sexual"
    mutation_applied: bool = False
    immigrant: bool = False
    status: str = "READY"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parent_ids = [pid for pid in [self.parent_a, self.parent_b] if pid]


__all__ = ["LineageRecord"]
