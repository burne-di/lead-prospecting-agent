"""Mock search tool: returns realistic *raw* results from local fixtures.

The fixtures deliberately mix valid businesses with noise (directories, news, job ads, other
cities, suppliers) so the agent has to qualify, not just copy. All data is fictitious.

Each fixture file is a "topic" (e.g. dental clinics in Lima). A query is routed to the topic
that matches it best and the documents of that topic are ranked by keyword overlap.
"""

from __future__ import annotations

import json
import re
from importlib import resources

from lead_agent.domain.models import normalize_text
from lead_agent.search.base import SearchResult

_STOPWORDS = {"de", "la", "el", "en", "y", "los", "las", "del", "para", "con", "a", "por", "peru"}
# Terms present in every topic: they rank documents but must not decide the topic.
_GENERIC = {"lima", "contac", "telefo", "empres", "servic", "negoci", "mejore", "email", "correo"}
# Minimal synonyms so natural queries reach the right topic ("dentista" -> dental).
_SYNONYMS = {"dentis": "dental", "odonto": "dental", "public": "market", "digita": "market"}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize_text(text))
    # crude stemming: "dentales"/"dental", "clinicas"/"clinica", "agencias"/"agencia"
    stems = {w[:6] for w in words if len(w) > 2 and w not in _STOPWORDS}
    return stems | {_SYNONYMS[s] for s in stems if s in _SYNONYMS}


class _Doc:
    def __init__(self, result: SearchResult) -> None:
        self.result = result
        self.title = _terms(result.title)
        self.body = _terms(f"{result.url} {result.content}") | self.title

    def score(self, q: set[str]) -> int:
        return len(q & self.body) + len(q & self.title)  # title matches weigh double


class MockSearch:
    def __init__(self, topics: dict[str, list[dict]] | None = None) -> None:
        topics = topics or self._load_fixtures()
        self._topics = {
            name: [_Doc(SearchResult(**d)) for d in docs] for name, docs in topics.items()
        }

    @staticmethod
    def _load_fixtures() -> dict[str, list[dict]]:
        folder = resources.files("lead_agent.search") / "fixtures"
        return {
            entry.name.removesuffix(".json"): json.loads(entry.read_text(encoding="utf-8"))
            for entry in sorted(folder.iterdir(), key=lambda p: p.name)
            if entry.name.endswith(".json")
        }

    def search(self, query: str, *, max_results: int = 6) -> list[SearchResult]:
        q = _terms(query)
        specific = q - _GENERIC
        best_topic, best = None, 0
        for name, docs in self._topics.items():
            # topic relevance = how many specific query terms the topic covers at all
            coverage = len(specific & set().union(*(d.body for d in docs)))
            if coverage > best:
                best_topic, best = name, coverage
        if best_topic is None:
            return []
        docs = self._topics[best_topic]
        ranked = sorted(
            ((d.score(q), -i, d) for i, d in enumerate(docs) if d.score(q) > 0),
            key=lambda s: (s[0], s[1]),
            reverse=True,
        )
        return [d.result for _, _, d in ranked[:max_results]]
