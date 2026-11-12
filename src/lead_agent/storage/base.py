"""Persistence port."""

from __future__ import annotations

from typing import Protocol

from lead_agent.domain import Lead, Run


class DuplicateLeadError(Exception):
    """A lead with the same lead_id already exists (atomic, conditional insert)."""


class LeadRepository(Protocol):
    def save_lead(self, lead: Lead) -> None:
        """Insert-if-absent. Raises DuplicateLeadError if lead_id already exists."""

    def get_lead(self, lead_id: str) -> Lead | None: ...

    def list_leads(self, *, sector: str | None = None, limit: int = 50) -> list[Lead]: ...

    def save_run(self, run: Run) -> None:
        """Upsert the run (audit record)."""

    def get_run(self, run_id: str) -> Run | None: ...

    def list_runs(self, *, limit: int = 20) -> list[Run]:
        """Most recent runs first."""
