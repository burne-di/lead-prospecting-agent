"""Search port."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class SearchError(RuntimeError):
    """The search provider failed (network, quota, auth...)."""


class SearchResult(BaseModel):
    url: str
    title: str
    content: str


class SearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 6) -> list[SearchResult]: ...
