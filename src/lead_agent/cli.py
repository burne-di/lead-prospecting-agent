"""CLI controller: `lead-agent run --sector ... --location ...` and `lead-agent list`."""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from lead_agent.config import get_settings
from lead_agent.container import build_service
from lead_agent.domain import Lead, Run, RunRequest
from lead_agent.logging_setup import configure_logging


def _print_run(run: Run, leads: list[Lead]) -> None:
    print(f"\nRun {run.run_id} · {run.status.value}")
    print(
        f"  pasos={run.steps} búsquedas={run.searches} guardados={len(leads)} "
        f"rechazos={run.rejected_saves} duplicados={run.duplicates} "
        f"tokens={run.usage.input_tokens}+{run.usage.output_tokens}"
    )
    if run.summary:
        print(f"  resumen: {run.summary}")
    for r in run.rejections:
        print(f"  rechazado: {r}")
    if run.error:
        print(f"  error: {run.error}")
    for lead in leads:
        c = lead.contact
        contact = ", ".join(c.emails + c.phones) or "sin contacto"
        print(f"\n  [{lead.fit_score}/5] {lead.name} — {lead.location or '?'}")
        print(f"        {lead.summary}")
        print(f"        {lead.website or '-'} · {contact}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lead-agent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Ejecuta el agente de prospección")
    p_run.add_argument("--sector", required=True, help='p. ej. "Clínicas dentales"')
    p_run.add_argument("--location", required=True, help='p. ej. "Lima, Perú"')
    p_run.add_argument("--max-leads", type=int, default=5)
    p_run.add_argument("--json", action="store_true", help="Salida JSON")

    p_list = sub.add_parser("list", help="Lista leads guardados")
    p_list.add_argument("--sector")
    p_list.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    settings = get_settings().model_copy(update={"run_mode": "sync"})  # CLI always runs inline
    configure_logging(
        settings.log_level,  # logs go to stderr, so --json stdout stays clean
        json_logs=settings.log_format == "json",
    )
    service = build_service(settings)

    if args.cmd == "list":
        for lead in service.list_leads(sector=args.sector, limit=args.limit):
            print(lead.model_dump_json(indent=2))
        return 0

    try:
        request = RunRequest(sector=args.sector, location=args.location, max_leads=args.max_leads)
    except ValidationError as exc:
        print(f"Parámetros inválidos: {exc}", file=sys.stderr)
        return 2
    run = service.start_run(request)
    leads = service.leads_of(run)
    if args.json:
        print(
            json.dumps(
                {
                    "run": run.model_dump(mode="json"),
                    "leads": [lead.model_dump(mode="json") for lead in leads],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_run(run, leads)
    return 0 if run.status.value != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
