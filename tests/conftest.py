from __future__ import annotations

import pytest

from lead_agent.agent.runner import LeadAgent
from lead_agent.domain import RunRequest
from lead_agent.search.base import SearchResult
from lead_agent.services.dispatch import InlineDispatcher
from lead_agent.services.prospecting import ProspectingService
from lead_agent.storage.sqlite import SQLiteLeadRepository

SRC_URL = "https://acme-dental.example/contacto"
SOURCE = SearchResult(
    url=SRC_URL,
    title="Acme Dental | Clínica dental en Miraflores, Lima",
    content=(
        "Acme Dental ofrece ortodoncia e implantes en Miraflores, Lima. "
        "Av. Pardo 610, Miraflores. Tel: +51 955 000 111. Email: citas@acme-dental.example. "
        "https://www.instagram.com/acmedental.example"
    ),
)
NOISE = SearchResult(
    url="https://directorio.example/dentistas",
    title="Directorio de dentistas en Lima",
    content="Anúnciate: ventas@directorio.example",
)


class FakeSearch:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results if results is not None else [SOURCE, NOISE]
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 6) -> list[SearchResult]:
        self.queries.append(query)
        return self.results[:max_results]


def lead_args(**overrides):
    base = dict(
        name="Acme Dental",
        sector="Clínicas dentales",
        summary="Clínica de ortodoncia e implantes en Miraflores.",
        website="https://acme-dental.example",
        location="Miraflores, Lima",
        contact=dict(
            emails=["citas@acme-dental.example"],
            phones=["955 000 111"],
            address="Av. Pardo 610, Miraflores",
            social_profiles=["https://www.instagram.com/acmedental.example"],
        ),
        source_urls=[SRC_URL],
        fit_score=5,
        fit_reason="Clínica dental en Lima con contacto.",
    )
    base.update(overrides)
    return base


@pytest.fixture
def request_() -> RunRequest:
    return RunRequest(sector="Clínicas dentales", location="Lima, Perú", max_leads=3)


@pytest.fixture
def repo() -> SQLiteLeadRepository:
    return SQLiteLeadRepository(":memory:")


@pytest.fixture
def make_service(repo):
    def factory(llm, search=None, **agent_kwargs) -> ProspectingService:
        return ProspectingService(
            repository=repo,
            agent_factory=lambda: LeadAgent(
                llm=llm, search=search or FakeSearch(), repository=repo, **agent_kwargs
            ),
            dispatcher=InlineDispatcher(),
        )

    return factory
