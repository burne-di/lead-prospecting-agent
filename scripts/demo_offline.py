"""Offline demo: the full pipeline without API keys.

Only the LLM decisions are scripted. The mock search tool, Pydantic validation, the grounding
guard, de-duplication and SQLite persistence are the real code. In the second turn the
"LLM" tries to save an email that is NOT in the source and the guard rejects it.

    python scripts/demo_offline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from lead_agent.agent.runner import LeadAgent
from lead_agent.domain import RunRequest
from lead_agent.llm.base import AssistantMessage, LLMResponse, Message, ToolResultsMessage
from lead_agent.llm.scripted import ScriptedLLM, call
from lead_agent.search.mock import MockSearch
from lead_agent.services.dispatch import InlineDispatcher
from lead_agent.services.prospecting import ProspectingService
from lead_agent.storage.sqlite import SQLiteLeadRepository

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SONRISA = "https://sonrisaplena-dental.example/contacto"
ORTODENT = "https://ortodent-losolivos.example/nosotros"
DENTALPRO = "https://dentalpro-surco.example"
DENTALKIDS = "https://www.facebook.com/dentalkids.sanborja.example"

sonrisa = dict(
    name="Clínica Dental Sonrisa Plena",
    sector="Clínicas dentales",
    summary="Clínica de odontología general, implantes dentales y blanqueamiento en Miraflores.",
    website="https://sonrisaplena-dental.example",
    location="Miraflores, Lima",
    contact=dict(
        # info@ is NOT in the source: a typical hallucination built from the domain
        emails=["citas@sonrisaplena-dental.example", "info@sonrisaplena-dental.example"],
        phones=["+51 955 000 142"],
        address="Av. José Larco 1234, Miraflores, Lima",
        social_profiles=["https://www.instagram.com/sonrisaplena.example"],
    ),
    source_urls=[SONRISA],
    fit_score=5,
    fit_reason="Clínica dental en Lima con teléfono, email y dirección.",
)
sonrisa_fixed = {
    **sonrisa,
    "contact": {**sonrisa["contact"], "emails": sonrisa["contact"]["emails"][:1]},
}

SCRIPT = [
    AssistantMessage(
        "Busco clínicas dentales en distritos de Lima con datos de contacto.",
        [call("search_web", query="clínica dental Lima contacto Miraflores Olivos")],
    ),
    AssistantMessage(
        "Descarto el directorio, el aviso de empleo y el laboratorio proveedor. "
        "Guardo Sonrisa Plena, OrtoDent y DentalPro.",
        [
            call("save_lead", **sonrisa),
            call(
                "save_lead",
                name="OrtoDent Los Olivos",
                sector="Ortodoncia",
                summary="Consultorio especializado en ortodoncia con brackets metálicos, "
                "estéticos y alineadores en Los Olivos.",
                website="https://ortodent-losolivos.example",
                location="Los Olivos, Lima",
                contact=dict(
                    emails=["contacto@ortodent-losolivos.example"],
                    phones=["(01) 555-0381"],
                    address="Av. Carlos Izaguirre 845, Los Olivos, Lima",
                ),
                source_urls=[ORTODENT],
                fit_score=5,
                fit_reason="Consultorio de ortodoncia en Lima Norte con contacto completo.",
            ),
            call(
                "save_lead",
                name="DentalPro Surco",
                sector="Odontología estética",
                summary="Ofrece odontología estética, carillas e implantes en Santiago de Surco.",
                website="https://dentalpro-surco.example",
                location="Santiago de Surco, Lima",
                contact=dict(
                    phones=["01 555 4410"], address="Av. Primavera 2050, Santiago de Surco"
                ),
                source_urls=[DENTALPRO],
                fit_score=5,
                fit_reason="Clínica dental en Lima con teléfono y dirección.",
            ),
        ],
    ),
    AssistantMessage(
        "El email info@ no estaba en la fuente: lo elimino y reintento. Busco odontopediatría.",
        [
            call("save_lead", **sonrisa_fixed),
            call("search_web", query="odontopediatría San Borja Lima"),
        ],
    ),
    AssistantMessage(
        "Dental Kids califica (solo tiene Facebook: va en social_profiles, sin website). "
        "El blog, el directorio y la noticia no son leads.",
        [
            call(
                "save_lead",
                name="Dental Kids San Borja",
                sector="Odontopediatría",
                summary="Atención dental para niños y adolescentes en San Borja.",
                location="San Borja, Lima",
                contact=dict(
                    phones=["987 555 120"],
                    address="Calle Las Begonias 310, San Borja",
                    social_profiles=[DENTALKIDS],
                ),
                source_urls=[DENTALKIDS],
                fit_score=5,
                fit_reason="Odontopediatría en Lima con teléfono y dirección.",
            )
        ],
    ),
    AssistantMessage(
        "",
        [
            call(
                "finish",
                summary="4 leads en Lima. Descartados: directorio, aviso de empleo, "
                "laboratorio proveedor, blog y noticia.",
            )
        ],
    ),
]


class NarratingLLM:
    """Prints what the (scripted) model thinks and the real tool results it receives."""

    def __init__(self, inner: ScriptedLLM) -> None:
        self._inner = inner

    def complete(self, *, system: str, messages: list[Message], tools) -> LLMResponse:
        if messages and isinstance(messages[-1], ToolResultsMessage):
            for r in messages[-1].results:
                p = json.loads(r.content)
                status = p["status"]
                detail = {
                    "ok": len(p.get("results", [])) if "results" in p else "",
                    "saved": p.get("lead_id"),
                    "duplicate": p.get("lead_id"),
                    "rejected": p.get("errors"),
                }.get(status, p.get("error", ""))
                print(f"   ← {status:<9} {detail}")
            print()
        resp = self._inner.complete(system=system, messages=messages, tools=tools)
        if resp.message.text:
            print(f"🧠 {resp.message.text}")
        for c in resp.message.tool_calls:
            arg = c.input.get("query") or c.input.get("name") or c.input.get("summary")
            print(f"   → {c.name}({arg!r})")
        return resp


def main() -> None:
    db = Path(tempfile.gettempdir()) / "lead_agent_demo.db"
    db.unlink(missing_ok=True)
    repo = SQLiteLeadRepository(db)
    service = ProspectingService(
        repository=repo,
        agent_factory=lambda: LeadAgent(
            llm=NarratingLLM(ScriptedLLM(SCRIPT)), search=MockSearch(), repository=repo
        ),
        dispatcher=InlineDispatcher(),
    )
    run = service.start_run(RunRequest(sector="Clínicas dentales", location="Lima, Perú"))
    leads = service.leads_of(run)
    print(
        f"\n✔ run {run.status.value}: {len(leads)} guardados, "
        f"{run.rejected_saves} rechazo(s) por grounding, "
        f"{run.searches} búsquedas, {run.steps} pasos"
    )
    print(f"  base de datos: {db}\n")
    print(leads[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
