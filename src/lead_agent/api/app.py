"""HTTP controller (FastAPI). Thin: validate input, call the service, map errors to HTTP."""

from __future__ import annotations

from importlib import resources

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from lead_agent import __version__
from lead_agent.config import get_settings
from lead_agent.container import get_service
from lead_agent.domain import Lead, Run, RunRequest
from lead_agent.services import ProspectingService, RunNotFoundError


class RunResponse(BaseModel):
    run: Run
    leads: list[Lead] = []


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lead Prospecting Agent",
        version=__version__,
        description="Agente IA que busca, cualifica y guarda leads por sector y ubicación.",
    )

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        """Minimal web UI to launch runs and inspect the agent's trace and leads."""
        page = resources.files("lead_agent.api").joinpath("static/index.html")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # AWS production architecture diagram, in the same three formats as docs/.
    _diagrams = {
        "svg": ("image/svg+xml", False),
        "png": ("image/png", True),
        "drawio": ("application/xml", True),  # draw.io source: plain mxfile XML
    }

    @app.get("/architecture.{fmt}", include_in_schema=False)
    def architecture(fmt: str) -> Response:
        if fmt not in _diagrams:
            raise HTTPException(status_code=404, detail="formato no disponible")
        media_type, download = _diagrams[fmt]
        data = resources.files("lead_agent.api").joinpath(f"static/architecture.{fmt}").read_bytes()
        headers = (
            {"Content-Disposition": f'attachment; filename="lead-agent-arquitectura-aws.{fmt}"'}
            if download
            else {}
        )
        return Response(data, media_type=media_type, headers=headers)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, str | int]:
        """Active configuration, for the UI header. Never exposes secrets."""
        s = get_settings()
        return {
            "version": __version__,
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model,
            "search_provider": s.search_provider,
            "storage_backend": s.storage_backend,
            "run_mode": s.run_mode,
            "max_searches": s.agent_max_searches,
            "max_steps": s.agent_max_steps,
        }

    @app.post("/runs", response_model=RunResponse)
    def create_run(
        body: RunRequest,
        response: Response,
        service: ProspectingService = Depends(get_service),
    ) -> RunResponse:
        run = service.start_run(body)
        if service.is_async:
            response.status_code = status.HTTP_202_ACCEPTED
            return RunResponse(run=run)
        return RunResponse(run=run, leads=service.leads_of(run))

    @app.get("/runs", response_model=list[Run])
    def list_runs(
        limit: int = Query(20, ge=1, le=100),
        service: ProspectingService = Depends(get_service),
    ) -> list[Run]:
        return service.list_runs(limit=limit)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str, service: ProspectingService = Depends(get_service)) -> RunResponse:
        try:
            run = service.get_run(run_id)
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="run no encontrado") from None
        return RunResponse(run=run, leads=service.leads_of(run))

    @app.get("/leads", response_model=list[Lead])
    def list_leads(
        sector: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        service: ProspectingService = Depends(get_service),
    ) -> list[Lead]:
        return service.list_leads(sector=sector, limit=limit)

    return app


app = create_app()
