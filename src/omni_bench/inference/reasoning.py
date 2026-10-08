"""Reasoning axis: how reasoning is separated from the answer in a response.

``as_is`` (default) — ``content`` is the answer as received.

Applied right after the response arrives, before an adapter's official parser
sees the text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from omni_bench.inference.base import BuildContext, Registry


class ReasoningStrategy(ABC):
    name: str

    @classmethod
    def create(cls, ctx: BuildContext) -> "ReasoningStrategy":
        return cls()

    @abstractmethod
    def split(self, content: str, server_reasoning: str | None) -> tuple[str, str | None]:
        """(answer text for the adapter parser, reasoning to record)."""
        raise NotImplementedError


REASONING_STRATEGIES: Registry[ReasoningStrategy] = Registry("reasoning")


@REASONING_STRATEGIES.register("as_is")
class AsIsReasoning(ReasoningStrategy):
    def split(self, content: str, server_reasoning: str | None) -> tuple[str, str | None]:
        return content, server_reasoning
