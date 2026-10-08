"""Reasoning axis: how reasoning is separated from the answer in a response.

``as_is`` (default) — ``content`` is the answer as received; a separate
                     reasoning field from the server (vLLM reasoning parser)
                     is recorded as ``reasoning``.
``split``          — the server returns everything in ``content``; text up to
                     the last ``</think>`` is reasoning, only what follows is
                     the answer.

Applied right after the response arrives (and by ``omni-bench rescore`` on
stored records), before an adapter's official parser sees the text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from omni_bench.inference.base import BuildContext, Registry

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


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


@REASONING_STRATEGIES.register("split")
class SplitReasoning(ReasoningStrategy):
    def split(self, content: str, server_reasoning: str | None) -> tuple[str, str | None]:
        cut = content.rfind(THINK_CLOSE)
        if cut < 0:
            return content, server_reasoning
        inline = content[:cut].strip()
        if inline.startswith(THINK_OPEN):
            inline = inline[len(THINK_OPEN):].strip()
        answer = content[cut + len(THINK_CLOSE):].strip()
        parts = [p for p in (server_reasoning, inline) if p]
        return answer, "\n\n".join(parts) if parts else None


def resplit_record(strategy: ReasoningStrategy, record: dict, field: str = "response") -> dict:
    """Re-apply ``strategy`` to a stored record in place (used by ``rescore``).

    The server content is ``response_raw`` if present, else ``record[field]``.
    When ``response_raw`` is present the stored reasoning already holds text
    split out of it, so only the server field is re-derived from scratch.
    """
    raw = record.get("response_raw")
    content = raw if raw is not None else (record.get(field) or "")
    server_reasoning = None if raw is not None else record.get("reasoning")
    text, reasoning = strategy.split(content, server_reasoning)
    if text != content:
        record[field] = text
        record["response_raw"] = content
    elif raw is not None:
        record[field] = text
        record.pop("response_raw")
    if reasoning != record.get("reasoning") and ("reasoning" in record or reasoning is not None):
        record["reasoning"] = reasoning
    return record
