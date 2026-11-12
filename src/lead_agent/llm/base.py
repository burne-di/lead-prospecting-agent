"""LLM port: provider-neutral conversation model + the interface every adapter implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from lead_agent.domain.models import TokenUsage


class LLMError(RuntimeError):
    """Unrecoverable LLM failure (after the adapter's own retries)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class UserMessage:
    text: str


@dataclass
class AssistantMessage:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolResultsMessage:
    results: list[ToolResult]


Message = UserMessage | AssistantMessage | ToolResultsMessage


@dataclass
class LLMResponse:
    message: AssistantMessage
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str | None = None


class LLMClient(Protocol):
    def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse: ...
