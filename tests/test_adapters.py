"""LLM and search adapters, with fake SDK clients (no network)."""

import json
from types import SimpleNamespace as NS

import httpx
import pytest

from lead_agent.llm.anthropic_client import ClaudeClient, to_anthropic_messages
from lead_agent.llm.base import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)
from lead_agent.llm.openai_client import OpenAICompatibleClient, to_openai_messages
from lead_agent.search.base import SearchError
from lead_agent.search.mock import MockSearch
from lead_agent.search.tavily import TavilySearch

CONVO = [
    UserMessage("tarea"),
    AssistantMessage("pienso", [ToolCall("t1", "search_web", {"query": "q"})]),
    ToolResultsMessage([ToolResult("t1", '{"status":"ok"}')]),
]
TOOLS = [ToolSpec("search_web", "busca", {"type": "object", "properties": {}})]


def test_anthropic_message_mapping():
    msgs = to_anthropic_messages(CONVO)
    assert msgs[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "search_web",
        "input": {"query": "q"},
    }
    assert msgs[2]["role"] == "user" and msgs[2]["content"][0]["tool_use_id"] == "t1"


def test_claude_client_parses_tool_use():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(
            content=[
                NS(type="text", text="ok"),
                NS(type="tool_use", id="x", name="finish", input={"summary": "s"}),
            ],
            usage=NS(input_tokens=10, output_tokens=5),
            stop_reason="tool_use",
        )

    fake = NS(messages=NS(create=create))
    resp = ClaudeClient(model="m", client=fake).complete(system="S", messages=CONVO, tools=TOOLS)
    assert resp.message.tool_calls[0].name == "finish" and resp.usage.input_tokens == 10
    assert captured["temperature"] == 0.0 and captured["tools"][0]["input_schema"]


def test_openai_message_mapping_and_parsing():
    msgs = to_openai_messages("S", CONVO)
    assert msgs[0]["role"] == "system" and msgs[3] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": '{"status":"ok"}',
    }

    def create(**kw):
        tc = NS(id="c1", function=NS(name="save_lead", arguments="{bad json"))
        return NS(
            choices=[NS(message=NS(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
            usage=NS(prompt_tokens=7, completion_tokens=3),
        )

    fake = NS(chat=NS(completions=NS(create=create)))
    resp = OpenAICompatibleClient(model="m", api_key="k", client=fake).complete(
        system="S", messages=CONVO, tools=TOOLS
    )
    assert resp.message.tool_calls[0].input == {"_raw_arguments": "{bad json"}


def test_mock_search_routes_by_topic():
    m = MockSearch()
    urls = [r.url for r in m.search("agencia de marketing digital Lima")]
    assert urls and all("dental" not in u for u in urls)
    assert m.search("zapaterías Cusco") == []
    assert m.search("Lima contacto") == []  # generic terms alone don't pick a topic
    assert "dental" in m.search("dentistas Lima Perú teléfono")[0].url  # synonym routing


def test_tavily_parses_and_retries(monkeypatch):
    monkeypatch.setattr("lead_agent.search.tavily.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        assert json.loads(request.content)["query"] == "q"
        return httpx.Response(
            200,
            json={"results": [{"url": "https://a.example", "title": "A", "content": "x" * 5000}]},
        )

    t = TavilySearch("key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    res = t.search("q")
    assert calls["n"] == 2 and len(res[0].content) == 3000


def test_tavily_raises_search_error_on_auth():
    t = TavilySearch(
        "key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key"))
        ),
    )
    with pytest.raises(SearchError):
        t.search("q")
