"""OpenAI-compatible adapter: GPT-4o-mini, Groq, Ollama, vLLM... (Chat Completions + tools)."""

from __future__ import annotations

import json
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


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.input)},
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolResultsMessage):
            out += [
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.results
            ]
    return out


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 4,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            if not api_key:
                raise LLMError("OPENAI_API_KEY no configurada")
            self._client = OpenAI(
                api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout
            )

    def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        import openai

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=to_openai_messages(system, messages),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    }
                    for t in tools
                ],
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Keep the call so the tool layer can answer with a validation error.
                args = {"_raw_arguments": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))
        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            message=AssistantMessage(text=(choice.content or "").strip(), tool_calls=calls),
            usage=usage,
            stop_reason=resp.choices[0].finish_reason,
        )
