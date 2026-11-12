"""Use cases. Shared by every controller (CLI, FastAPI, Lambda API, SQS worker)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from lead_agent.agent.runner import LeadAgent
from lead_agent.domain import Lead, Run, RunRequest, RunStatus
from lead_agent.domain.models import utcnow
from lead_agent.llm.base import LLMError
from lead_agent.search.base import SearchError
from lead_agent.services.dispatch import Dispatcher
from lead_agent.storage.base import LeadRepository

log = logging.getLogger(__name__)


class RunNotFoundError(LookupError):
    pass


class ProspectingService:
    def __init__(
        self,
        *,
        repository: LeadRepository,
        agent_factory: Callable[[], LeadAgent],
        dispatcher: Dispatcher,
    ) -> None:
        self._repo = repository
        self._agent_factory = agent_factory
        self._agent: LeadAgent | None = None
        self._dispatcher = dispatcher

    @property
    def is_async(self) -> bool:
        return self._dispatcher.is_async

    def start_run(self, request: RunRequest) -> Run:
        """Create the run record and hand it to the dispatcher (inline or queue)."""
        run = Run(request=request)
        self._repo.save_run(run)
        log.info("run created", extra={"run_id": run.run_id, "sector": request.sector})
        return self._dispatcher.dispatch(run, self.execute_run)

    def execute_run(self, run_id: str) -> Run:
        """Execute a queued run. Idempotent: a run already processed is not executed again."""
        run = self._repo.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        if run.status != RunStatus.QUEUED:
            log.warning("run already processed, skipping", extra={"run_id": run_id})
            return run

        run.status = RunStatus.RUNNING
        self._repo.save_run(run)
        try:
            if self._agent is None:
                self._agent = self._agent_factory()
        except (LLMError, SearchError) as exc:  # e.g. missing API key: fail the run, not the app
            run.status, run.error, run.finished_at = RunStatus.FAILED, str(exc), utcnow()
            self._repo.save_run(run)
            log.error("agent setup failed", extra={"run_id": run_id, "error": str(exc)})
            return run
        run, leads = self._agent.run(run)
        self._repo.save_run(run)
        log.info(
            "run finished",
            extra={"run_id": run.run_id, "status": run.status.value, "leads": len(leads)},
        )
        return run

    def get_run(self, run_id: str) -> Run:
        run = self._repo.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    def list_runs(self, *, limit: int = 20) -> list[Run]:
        return self._repo.list_runs(limit=limit)

    def leads_of(self, run: Run) -> list[Lead]:
        return [lead for i in run.lead_ids if (lead := self._repo.get_lead(i))]

    def list_leads(self, *, sector: str | None = None, limit: int = 50) -> list[Lead]:
        return self._repo.list_leads(sector=sector, limit=limit)
