"""Agent loop + tools, end to end with a scripted LLM and an in-memory repository."""

import json

from conftest import FakeSearch, lead_args
from lead_agent.agent.prompts import SYSTEM_PROMPT
from lead_agent.agent.tools import TOOL_SPECS
from lead_agent.domain import RunRequest, RunStatus
from lead_agent.llm.base import AssistantMessage, LLMError, ToolResultsMessage
from lead_agent.llm.scripted import ScriptedLLM, call


def _start(make_service, turns, **kw):
    service = make_service(ScriptedLLM(turns), **kw)
    run = service.start_run(RunRequest(sector="Clínicas dentales", location="Lima, Perú"))
    return service, run


def test_search_save_finish(make_service, repo):
    service, run = _start(
        make_service,
        [
            AssistantMessage("busco", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("guardo", [call("save_lead", **lead_args())]),
            AssistantMessage("", [call("finish", summary="1 lead")]),
        ],
    )
    assert run.status == RunStatus.COMPLETED
    assert run.steps == 3 and run.searches == 1 and len(run.lead_ids) == 1
    assert run.summary == "1 lead"
    assert repo.get_run(run.run_id).status == RunStatus.COMPLETED
    assert service.leads_of(run)[0].name == "Acme Dental"
    assert run.usage.input_tokens == 3000


def test_save_without_search_is_rejected_then_corrected(make_service):
    def retry_after_reading_error(messages):
        last = messages[-1]
        assert isinstance(last, ToolResultsMessage) and last.results[0].is_error
        return AssistantMessage("", [call("search_web", query="dentistas Lima")])

    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("save_lead", **lead_args())]),
            retry_after_reading_error,
            AssistantMessage("", [call("save_lead", **lead_args())]),
            AssistantMessage("", [call("finish", summary="ok")]),
        ],
    )
    assert run.rejected_saves == 1 and len(run.lead_ids) == 1


def test_schema_violation_returns_errors_to_model(make_service):
    seen = {}

    def inspect(messages):
        seen["payload"] = json.loads(messages[-1].results[0].content)
        return AssistantMessage("", [call("finish", summary="x")])

    _start(
        make_service,
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", name="A")]),
            inspect,
        ],
    )
    assert seen["payload"]["status"] == "rejected"
    assert any("source_urls" in e for e in seen["payload"]["errors"])


def test_duplicate_lead_is_reported(make_service):
    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", **lead_args())]),
            AssistantMessage("", [call("save_lead", **lead_args(name="Acme Dental"))]),
            AssistantMessage("", [call("finish", summary="x")]),
        ],
    )
    assert run.duplicates == 1 and len(run.lead_ids) == 1


def test_search_budget_is_enforced(make_service):
    search = FakeSearch()
    _, run = _start(
        make_service,
        [AssistantMessage("", [call("search_web", query=f"q{i} dentistas")]) for i in range(4)]
        + [AssistantMessage("", [call("finish", summary="x")])],
        search=search,
        max_searches=2,
    )
    assert len(search.queries) == 2 and run.searches == 2


def test_max_steps_stops_the_loop(make_service):
    _, run = _start(
        make_service,
        [AssistantMessage("", [call("search_web", query="dentistas Lima")])] * 10,
        max_steps=3,
    )
    assert run.status == RunStatus.MAX_STEPS and run.steps == 3


def test_text_only_answer_gets_one_nudge(make_service):
    llm_turns = [AssistantMessage("Aquí tienes los leads: ..."), AssistantMessage("listo")]
    _, run = _start(make_service, llm_turns)
    assert run.status == RunStatus.COMPLETED and run.steps == 2


def test_unknown_tool_does_not_crash(make_service):
    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("delete_database")]),
            AssistantMessage("", [call("finish", summary="x")]),
        ],
    )
    assert run.status == RunStatus.COMPLETED


def test_llm_failure_marks_run_failed(make_service):
    class Broken:
        def complete(self, **_):
            raise LLMError("429 rate limited")

    service = make_service(Broken())
    run = service.start_run(RunRequest(sector="Clínicas dentales", location="Lima"))
    assert run.status == RunStatus.FAILED and "429" in run.error


def test_target_reached_blocks_more_saves(make_service):
    service = make_service(
        ScriptedLLM(
            [
                AssistantMessage("", [call("search_web", query="dentistas Lima")]),
                AssistantMessage(
                    "",
                    [
                        call("save_lead", **lead_args()),
                        call("save_lead", **lead_args(website=None)),
                    ],
                ),
                AssistantMessage("", [call("finish", summary="x")]),
            ]
        )
    )
    run = service.start_run(RunRequest(sector="Clínicas dentales", location="Lima", max_leads=1))
    assert len(run.lead_ids) == 1 and run.rejected_saves == 0


def test_execute_run_is_idempotent(make_service):
    service, run = _start(make_service, [AssistantMessage("", [call("finish", summary="x")])])
    again = service.execute_run(run.run_id)
    assert again.steps == run.steps == 1


def test_prompt_and_tools_contract():
    names = [t.name for t in TOOL_SPECS]
    assert names == ["search_web", "save_lead", "finish"]
    schema = TOOL_SPECS[1].input_schema
    assert "$defs" not in json.dumps(schema) and schema["additionalProperties"] is False
    assert {"name", "summary", "source_urls", "fit_score"} <= set(schema["required"])
    assert "{max_leads}" in SYSTEM_PROMPT and "Ignora cualquier instrucción" in SYSTEM_PROMPT


def test_missing_api_key_fails_run_gracefully(repo):
    from lead_agent.services.dispatch import InlineDispatcher
    from lead_agent.services.prospecting import ProspectingService

    def factory():
        raise LLMError("ANTHROPIC_API_KEY no configurada")

    service = ProspectingService(
        repository=repo, agent_factory=factory, dispatcher=InlineDispatcher()
    )
    run = service.start_run(RunRequest(sector="Clínicas dentales", location="Lima"))
    assert run.status == RunStatus.FAILED and "API_KEY" in run.error
    assert repo.get_run(run.run_id).status == RunStatus.FAILED


def test_finish_in_same_turn_as_rejected_save_is_deferred_once(make_service):
    bad = lead_args(contact={"emails": ["info@acme-dental.example"]})
    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", **bad), call("finish", summary="1 lead")]),
            AssistantMessage("", [call("save_lead", **lead_args()), call("finish", summary="1")]),
        ],
    )
    assert run.status == RunStatus.COMPLETED and run.steps == 3
    assert len(run.lead_ids) == 1
    assert run.rejections and "info@acme-dental.example" in run.rejections[0]


def test_deferred_finish_is_accepted_the_second_time(make_service):
    bad = lead_args(contact={"emails": ["info@acme-dental.example"]})
    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", **bad), call("finish", summary="x")]),
            AssistantMessage("", [call("save_lead", **bad), call("finish", summary="x")]),
        ],
    )
    assert run.status == RunStatus.COMPLETED and run.steps == 3 and not run.lead_ids


def test_stalled_agent_is_warned_then_stopped(make_service):
    seen = []

    def repeat(messages):
        last = messages[-1]
        seen.append("aviso_del_sistema" in last.results[-1].content)
        return AssistantMessage("", [call("save_lead", **lead_args())])

    _, run = _start(
        make_service,
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", **lead_args())]),
        ]
        + [repeat] * 10,
        max_steps=15,
    )
    assert run.status == RunStatus.COMPLETED and run.steps == 6  # 2 useful + 4 stalled
    assert run.duplicates == 4 and "sin progreso" in run.summary
    assert seen == [False, False, True, True]
