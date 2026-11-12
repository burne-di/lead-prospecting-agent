"""ReAct loop: the LLM reasons, picks a tool, observes the result, and repeats until `finish`."""

from __future__ import annotations

import json
import logging

from lead_agent.agent.prompts import NUDGE_PROMPT, SYSTEM_PROMPT, TASK_PROMPT
from lead_agent.agent.tools import TOOL_SPECS, ToolContext, execute_tool
from lead_agent.domain import Lead, Run, RunStatus
from lead_agent.domain.models import TraceCall, TraceStep, utcnow
from lead_agent.llm.base import (
    LLMClient,
    LLMError,
    Message,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
)
from lead_agent.search.base import SearchProvider
from lead_agent.storage.base import LeadRepository

log = logging.getLogger(__name__)

# Loop guard: steps in a row without progress (no new lead saved, no successful search).
STALL_WARNING_STEPS = 2
STALL_STOP_STEPS = 4
STALL_WARNING = (
    "No estás avanzando: los últimos pasos no guardaron leads nuevos ni hicieron búsquedas "
    "útiles. No repitas llamadas. Haz una búsqueda distinta o llama a finish ahora."
)


def _with_warning(result: ToolResult, warning: str) -> ToolResult:
    """Attach a system warning to a tool result (keeps the tool_use/tool_result pairing)."""
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        payload = {"content": result.content}
    payload["aviso_del_sistema"] = warning
    return ToolResult(
        call_id=result.call_id,
        content=json.dumps(payload, ensure_ascii=False),
        is_error=result.is_error,
    )


def _trace_call(call: ToolCall, result: ToolResult) -> TraceCall:
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        payload = {"status": "error", "error": result.content}
    status = payload.get("status", "error")
    if "results" in payload:
        urls = [r.get("url", "") for r in payload["results"]]
        detail = f"{len(urls)} resultados: " + ", ".join(urls)
    elif status == "rejected":
        detail = "; ".join(payload.get("errors", []))
    elif status in {"saved", "duplicate"}:
        detail = f"lead_id={payload.get('lead_id')}"
    else:
        detail = payload.get("error") or payload.get("message") or ""
    args = json.dumps(call.input, ensure_ascii=False)
    return TraceCall(
        tool=call.name,
        input=args if len(args) <= 1500 else args[:1500] + "…",
        status=status,
        detail=detail[:1000],
    )


class LeadAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        search: SearchProvider,
        repository: LeadRepository,
        max_steps: int = 15,
        max_searches: int = 5,
        search_max_results: int = 6,
    ) -> None:
        self._llm = llm
        self._search = search
        self._repo = repository
        self._max_steps = max_steps
        self._max_searches = max_searches
        self._search_max_results = search_max_results

    def run(self, run: Run) -> tuple[Run, list[Lead]]:
        req = run.request
        ctx = ToolContext(
            request=req,
            run=run,
            repository=self._repo,
            search=self._search,
            max_searches=self._max_searches,
            search_max_results=self._search_max_results,
        )
        system = SYSTEM_PROMPT.format(max_leads=req.max_leads, max_searches=self._max_searches)
        messages: list[Message] = [
            UserMessage(
                TASK_PROMPT.format(
                    sector=req.sector, location=req.location, max_leads=req.max_leads
                )
            )
        ]
        run.status = RunStatus.RUNNING
        nudged = False
        finish_deferred = False  # finish is deferred at most once per run
        stopped = False  # True when the model ended the run itself (finish or explicit end_turn)
        stalled = 0

        try:
            while run.steps < self._max_steps:
                run.steps += 1
                resp = self._llm.complete(system=system, messages=messages, tools=TOOL_SPECS)
                run.usage.add(resp.usage)
                messages.append(resp.message)

                calls = resp.message.tool_calls
                log.info(
                    "agent step",
                    extra={
                        "run_id": run.run_id,
                        "step": run.steps,
                        "tools": [c.name for c in calls],
                        "thought": resp.message.text[:200],
                    },
                )

                step = TraceStep(step=run.steps, thought=resp.message.text[:600])
                run.trace.append(step)

                if not calls:
                    if nudged:  # the model insists on stopping without `finish`
                        stopped = True
                        break
                    nudged = True
                    messages.append(UserMessage(NUDGE_PROMPT))
                    continue

                # Every tool_use in a turn must get its tool_result in the next message.
                results = self._execute_turn(ctx, calls, allow_premature_finish=finish_deferred)
                finish_deferred = finish_deferred or ctx.finish_deferred
                step.calls = [_trace_call(c, r) for c, r in zip(calls, results, strict=True)]
                progress = any(
                    c.status == "saved" or (c.tool == "search_web" and c.status == "ok")
                    for c in step.calls
                )
                stalled = 0 if progress else stalled + 1
                if stalled >= STALL_STOP_STEPS and not ctx.finished:
                    log.warning("agent stalled, stopping", extra={"run_id": run.run_id})
                    run.summary = run.summary or (
                        f"Detenido por el sistema tras {stalled} pasos sin progreso; "
                        f"{len(ctx.saved)} leads guardados."
                    )
                    messages.append(ToolResultsMessage(results))
                    stopped = True
                    break
                if stalled >= STALL_WARNING_STEPS:
                    results[-1] = _with_warning(results[-1], STALL_WARNING)
                messages.append(ToolResultsMessage(results))
                for tc in step.calls:
                    log.info(
                        "tool call",
                        extra={
                            "run_id": run.run_id,
                            "step": run.steps,
                            "tool": tc.tool,
                            "status": tc.status,
                            "detail": tc.detail[:300],
                        },
                    )
                if ctx.finished:
                    stopped = True
                    break

            run.status = RunStatus.COMPLETED if stopped else RunStatus.MAX_STEPS
        except LLMError as exc:
            log.error("llm failure", extra={"run_id": run.run_id, "error": str(exc)})
            run.status = RunStatus.FAILED
            run.error = str(exc)

        run.finished_at = utcnow()
        return run, ctx.saved

    @staticmethod
    def _execute_turn(
        ctx: ToolContext, calls: list[ToolCall], *, allow_premature_finish: bool
    ) -> list[ToolResult]:
        """Run every tool call of the turn; `finish` goes last.

        Models often call save_lead(s) and finish in the same turn, so they never see a
        rejection they could fix. If any save of the turn was rejected, finish is answered with
        an error (once per run) so the model can correct it or explicitly confirm finish.
        """
        others = [c for c in calls if c.name != "finish"]
        finishes = [c for c in calls if c.name == "finish"]
        results = {c.id: execute_tool(ctx, c) for c in others}
        rejected = any(r.is_error and '"rejected"' in r.content for r in results.values())
        for c in finishes:
            if rejected and not allow_premature_finish and not ctx.target_reached:
                ctx.finish_deferred = True
                payload = {
                    "status": "error",
                    "error": "finish ignorado: hay save_lead rechazados en este turno. Corrige "
                    "lo indicado y reintenta, o llama a finish de nuevo si no es posible.",
                }
                results[c.id] = ToolResult(
                    call_id=c.id, content=json.dumps(payload, ensure_ascii=False), is_error=True
                )
            else:
                results[c.id] = execute_tool(ctx, c)
        return [results[c.id] for c in calls]  # keep the model's order
