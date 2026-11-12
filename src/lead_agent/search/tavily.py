"""Tavily adapter: returns the relevant content of each page, not just a snippet."""

from __future__ import annotations

import logging
import time

import httpx

from lead_agent.search.base import SearchError, SearchResult

log = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}


class TavilySearch:
    URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None,
        *,
        max_content_chars: int = 3000,
        timeout: float = 20.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise SearchError("TAVILY_API_KEY no configurada")
        self._api_key = api_key
        self._max_content_chars = max_content_chars
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, *, max_results: int = 6) -> list[SearchResult]:
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(self.URL, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == self._max_retries:
                    raise SearchError(f"error de red: {exc}") from exc
            else:
                if resp.status_code == 200:
                    break
                if resp.status_code not in _RETRYABLE or attempt == self._max_retries:
                    raise SearchError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(min(2**attempt, 8))  # exponential backoff for rate limits / 5xx

        data = resp.json()
        return [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", "") or "",
                content=(r.get("raw_content") or r.get("content") or "")[: self._max_content_chars],
            )
            for r in data.get("results", [])
            if r.get("url")
        ]
