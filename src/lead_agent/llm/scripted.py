"""Deterministic fake LLM for the offline demo and tests: replays a script of turns."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from lead_agent.domain.models import TokenUsage
from lead_agent.llm.base import AssistantMessage, LLMResponse, Message, ToolCall, ToolSpec

# A turn is either a fixed AssistantMessage or a function of the conversation so far
# (so a script can react to tool results, e.g. after a grounding rejection).
Turn = AssistantMessage | Callable[[list[Message]], AssistantMessage]

_ids = itertools.count(1)


def call(tool: str, /, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=tool, input=args)


class ScriptedLLM:
    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = list(turns)
        self.calls = 0
        self.last_system: str | None = None
        self.last_tools: list[ToolSpec] = []

    def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        self.last_system, self.last_tools = system, tools
        if self.calls >= len(self._turns):
            msg = AssistantMessage(text="(sin más turnos en el guion)")
        else:
            turn = self._turns[self.calls]
            msg = turn(messages) if callable(turn) else turn
        self.calls += 1
        return LLMResponse(message=msg, usage=TokenUsage(input_tokens=1000, output_tokens=150))
