"""Tools exposed to the LLM. Each tool = JSON schema (contract) + handler (side effect).

The LLM decides *when* to call each one; the handlers enforce the business rules
(budget of searches, schema validation, grounding, de-duplication, lead target).
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from lead_agent.agent.grounding import EvidenceStore, check_grounding
from lead_agent.domain import Lead, LeadDraft, Run, RunRequest
from lead_agent.llm.base import ToolCall, ToolResult, ToolSpec
from lead_agent.search.base import SearchError, SearchProvider
from lead_agent.storage.base import DuplicateLeadError, LeadRepository

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Mutable per-run state shared by the tool handlers."""

    request: RunRequest
    run: Run
    repository: LeadRepository
    search: SearchProvider
    max_searches: int
    search_max_results: int
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    saved: list[Lead] = field(default_factory=list)
    finished: bool = False
    finish_deferred: bool = False

    @property
    def target_reached(self) -> bool:
        return len(self.saved) >= self.request.max_leads


# --------------------------------------------------------------------------- schemas


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve Pydantic `$defs`/`$ref` so every provider receives a flat JSON Schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = walk(defs[node["$ref"].split("/")[-1]])
                siblings = {k: walk(v) for k, v in node.items() if k not in {"$ref", "title"}}
                return {**target, **siblings}
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(n) for n in node]
        return node

    resolved = walk(schema)
    resolved.pop("description", None)  # internal docstring, not meant for the LLM
    return resolved


SEARCH_WEB = ToolSpec(
    name="search_web",
    description=(
        "Busca en internet y devuelve resultados crudos (url, title, content). Úsala para "
        "descubrir negocios del sector y ubicación pedidos. Cada llamada consume presupuesto."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Consulta concreta, p. ej. 'dentista Miraflores Lima contacto'",
                "minLength": 3,
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

SAVE_LEAD = ToolSpec(
    name="save_lead",
    description=(
        "Guarda UN lead cualificado en la base de datos. Todos los datos deben aparecer "
        "literalmente en los resultados citados en source_urls; si no, se rechaza con el detalle "
        "de qué corregir."
    ),
    input_schema=_inline_refs(LeadDraft.model_json_schema()),
)

FINISH = ToolSpec(
    name="finish",
    description="Termina la ejecución. Llámala al alcanzar el objetivo o cuando no haya más leads.",
    input_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "1-3 frases: qué se encontró y por qué se descartó el resto.",
            }
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
)

TOOL_SPECS: list[ToolSpec] = [SEARCH_WEB, SAVE_LEAD, FINISH]


# --------------------------------------------------------------------------- handlers


def _search_web(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.run.searches >= ctx.max_searches:
        return {
            "status": "error",
            "error": "Presupuesto de búsquedas agotado. Guarda lo que tengas o llama a finish.",
        }
    query = str(args.get("query", "")).strip()
    if len(query) < 3:
        return {"status": "error", "error": "query vacía o demasiado corta"}

    ctx.run.searches += 1
    try:
        results = ctx.search.search(query, max_results=ctx.search_max_results)
    except SearchError as exc:
        log.warning("search failed", extra={"query": query, "error": str(exc)})
        return {"status": "error", "error": f"La búsqueda falló: {exc}. Prueba otra consulta."}

    ctx.evidence.add(results)
    return {
        "status": "ok",
        "query": query,
        "searches_left": ctx.max_searches - ctx.run.searches,
        "results": [r.model_dump() for r in results],
    }


def _save_lead(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    if ctx.target_reached:
        return {"status": "rejected", "errors": ["Objetivo de leads alcanzado. Llama a finish."]}

    try:
        draft = LeadDraft.model_validate(args)
    except ValidationError as exc:
        ctx.run.rejected_saves += 1
        errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        ctx.run.rejections.append(f"{args.get('name', '?')}: {'; '.join(errors)}")
        return {"status": "rejected", "errors": errors}

    errors = check_grounding(draft, ctx.evidence)
    if errors:
        ctx.run.rejected_saves += 1
        ctx.run.rejections.append(f"{draft.name}: {'; '.join(errors)}")
        log.info("lead rejected by grounding", extra={"lead": draft.name, "errors": errors})
        return {"status": "rejected", "errors": errors}

    lead = Lead.from_draft(draft, run_id=ctx.run.run_id, request=ctx.request)
    try:
        ctx.repository.save_lead(lead)
    except DuplicateLeadError:
        ctx.run.duplicates += 1
        where = (
            "en esta ejecución"
            if lead.lead_id in ctx.run.lead_ids
            else "por una ejecución anterior"
        )
        return {
            "status": "duplicate",
            "lead_id": lead.lead_id,
            "message": f"'{lead.name}' ya está guardado {where}: NO lo vuelvas a guardar.",
            "leads_saved_this_run": len(ctx.saved),
            "searches_left": ctx.max_searches - ctx.run.searches,
            "next": "Busca negocios distintos con otra consulta, o llama a finish si no quedan.",
        }

    ctx.saved.append(lead)
    ctx.run.lead_ids.append(lead.lead_id)
    response: dict[str, Any] = {
        "status": "saved",
        "lead_id": lead.lead_id,
        "leads_saved": len(ctx.saved),
        "target": ctx.request.max_leads,
    }
    if ctx.target_reached:
        response["next"] = "Objetivo alcanzado: llama a finish."
    return response


def _finish(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    ctx.finished = True
    ctx.run.summary = str(args.get("summary", "")).strip() or None
    return {"status": "ok", "leads_saved": len(ctx.saved)}


_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    SEARCH_WEB.name: _search_web,
    SAVE_LEAD.name: _save_lead,
    FINISH.name: _finish,
}


def execute_tool(ctx: ToolContext, call: ToolCall) -> ToolResult:
    """Run one tool call. Never raises: failures are returned to the LLM as tool errors."""
    handler = _HANDLERS.get(call.name)
    if handler is None:
        payload: dict[str, Any] = {
            "status": "error",
            "error": f"Herramienta desconocida: {call.name}",
        }
    else:
        try:
            payload = handler(ctx, call.input)
        except Exception as exc:  # defensive: a tool bug must not kill the run
            log.exception("tool crashed", extra={"tool": call.name})
            payload = {"status": "error", "error": f"Error interno en {call.name}: {exc}"}
    is_error = payload.get("status") in {"error", "rejected"}
    return ToolResult(
        call_id=call.id, content=json.dumps(payload, ensure_ascii=False), is_error=is_error
    )
