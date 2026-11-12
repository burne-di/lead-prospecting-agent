"""Claude adapter. The same Messages API serves both the Anthropic API and Amazon Bedrock."""

from __future__ import annotations

from typing import Any

from lead_agent.domain.models import TokenUsage
from lead_agent.llm.base import (
    AssistantMessage,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            if m.text:
                blocks.append({"type": "text", "text": m.text})
            blocks += [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                for c in m.tool_calls
            ]
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
        elif isinstance(m, ToolResultsMessage):
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in m.results
                    ],
                }
            )
    return out


class ClaudeClient:
    def __init__(
        self,
        *,
        model: str,
        provider: str = "anthropic",
        api_key: str | None = None,
        aws_region: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_retries: int = 4,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        if client is not None:
            self._client = client
        elif provider == "bedrock":
            from anthropic import AnthropicBedrock

            # Credentials come from the standard AWS chain (IAM role on Lambda).
            self._client = AnthropicBedrock(
                aws_region=aws_region, max_retries=max_retries, timeout=timeout
            )
        else:
            from anthropic import Anthropic

            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY no configurada")
            self._client = Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)

    def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        import anthropic

        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=to_anthropic_messages(messages),
                tools=[
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in tools
                ],
            )
        except anthropic.APIError as exc:  # SDK already retried 429/5xx with backoff
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))
        return LLMResponse(
            message=AssistantMessage(text="\n".join(text_parts).strip(), tool_calls=calls),
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens
            ),
            stop_reason=resp.stop_reason,
        )
