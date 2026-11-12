"""Composition root: the only place that knows the concrete implementations."""

from __future__ import annotations

from functools import lru_cache

from lead_agent.agent.runner import LeadAgent
from lead_agent.config import Settings, get_settings, secret
from lead_agent.llm.base import LLMClient
from lead_agent.search.base import SearchProvider
from lead_agent.services.dispatch import Dispatcher, InlineDispatcher, SQSDispatcher
from lead_agent.services.prospecting import ProspectingService
from lead_agent.storage.base import LeadRepository


def build_llm(s: Settings) -> LLMClient:
    if s.llm_provider == "openai":
        from lead_agent.llm.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model=s.llm_model,
            api_key=secret(s.openai_api_key),
            base_url=s.openai_base_url,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
    from lead_agent.llm.anthropic_client import ClaudeClient

    return ClaudeClient(
        model=s.llm_model,
        provider=s.llm_provider,
        api_key=secret(s.anthropic_api_key),
        aws_region=s.aws_region,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )


def build_search(s: Settings) -> SearchProvider:
    if s.search_provider == "tavily":
        from lead_agent.search.tavily import TavilySearch

        return TavilySearch(secret(s.tavily_api_key))
    from lead_agent.search.mock import MockSearch

    return MockSearch()


def build_repository(s: Settings) -> LeadRepository:
    if s.storage_backend == "dynamodb":
        from lead_agent.storage.dynamodb import DynamoDBLeadRepository

        return DynamoDBLeadRepository(
            leads_table=s.dynamodb_leads_table,
            runs_table=s.dynamodb_runs_table,
            region=s.aws_region,
            endpoint_url=s.dynamodb_endpoint_url,
        )
    from lead_agent.storage.sqlite import SQLiteLeadRepository

    return SQLiteLeadRepository(s.sqlite_path)


def build_service(
    s: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    search: SearchProvider | None = None,
    repository: LeadRepository | None = None,
) -> ProspectingService:
    s = s or get_settings()
    repo = repository or build_repository(s)

    def agent_factory() -> LeadAgent:  # lazy: the API Lambda never needs an LLM client
        return LeadAgent(
            llm=llm or build_llm(s),
            search=search or build_search(s),
            repository=repo,
            max_steps=s.agent_max_steps,
            max_searches=s.agent_max_searches,
            search_max_results=s.search_max_results,
        )

    dispatcher: Dispatcher
    if s.run_mode == "async":
        if not s.runs_queue_url:
            raise ValueError("RUN_MODE=async requiere RUNS_QUEUE_URL")
        dispatcher = SQSDispatcher(s.runs_queue_url, region=s.aws_region)
    else:
        dispatcher = InlineDispatcher()
    return ProspectingService(repository=repo, agent_factory=agent_factory, dispatcher=dispatcher)


@lru_cache
def get_service() -> ProspectingService:
    """Process-wide singleton (reused across Lambda invocations / API requests)."""
    return build_service()
