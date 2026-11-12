"""FastAPI, CLI and Lambda controllers on top of the same service."""

import json

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from conftest import lead_args
from lead_agent.api.app import create_app
from lead_agent.container import get_service
from lead_agent.domain import RunStatus
from lead_agent.llm.base import AssistantMessage
from lead_agent.llm.scripted import ScriptedLLM, call
from lead_agent.services.dispatch import SQSDispatcher


def _script():
    return ScriptedLLM(
        [
            AssistantMessage("", [call("search_web", query="dentistas Lima")]),
            AssistantMessage("", [call("save_lead", **lead_args())]),
            AssistantMessage("", [call("finish", summary="1 lead")]),
        ]
    )


@pytest.fixture
def client(make_service):
    app = create_app()
    service = make_service(_script())
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_post_run_sync_returns_leads(client):
    r = client.post("/runs", json={"sector": "Clínicas dentales", "location": "Lima, Perú"})
    assert r.status_code == 200
    body = r.json()
    assert body["run"]["status"] == "completed" and body["leads"][0]["name"] == "Acme Dental"

    r2 = client.get(f"/runs/{body['run']['run_id']}")
    assert r2.status_code == 200 and len(r2.json()["leads"]) == 1
    assert len(client.get("/leads", params={"sector": "Clínicas dentales"}).json()) == 1


def test_web_ui_and_run_history(client):
    page = client.get("/")
    assert page.status_code == 200 and "Lead Prospecting Agent" in page.text
    client.post("/runs", json={"sector": "Clínicas dentales", "location": "Lima, Perú"})
    runs = client.get("/runs").json()
    assert len(runs) == 1
    trace = runs[0]["trace"]
    assert [c["tool"] for step in trace for c in step["calls"]] == [
        "search_web",
        "save_lead",
        "finish",
    ]
    assert trace[0]["calls"][0]["detail"].startswith("2 resultados")
    assert trace[1]["calls"][0]["status"] == "saved"


def test_info_exposes_config_but_no_secrets(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    from lead_agent.config import get_settings

    get_settings.cache_clear()
    try:
        body = client.get("/info").json()
    finally:
        get_settings.cache_clear()
    assert {"llm_model", "search_provider", "storage_backend"} <= body.keys()
    assert "sk-test-secret" not in str(body) and not any("key" in k for k in body)


def test_architecture_diagram_is_served_and_matches_docs(client):
    from pathlib import Path

    r = client.get("/architecture.svg")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
    assert "Amazon Bedrock" in r.text and "Amazon DynamoDB" in r.text
    docs = Path(__file__).parents[1] / "docs" / "architecture.svg"
    assert docs.read_text(encoding="utf-8") == r.text  # one diagram, two places


def test_validation_and_404(client):
    assert client.post("/runs", json={"sector": "x"}).status_code == 422
    assert client.get("/runs/missing").status_code == 404


def test_async_mode_enqueues_and_worker_executes(make_service, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")
        url = sqs.create_queue(QueueName="runs")["QueueUrl"]
        service = make_service(_script())
        service._dispatcher = SQSDispatcher(url, client=sqs)

        app = create_app()
        app.dependency_overrides[get_service] = lambda: service
        r = TestClient(app).post("/runs", json={"sector": "Clínicas dentales", "location": "Lima"})
        assert r.status_code == 202 and r.json()["run"]["status"] == "queued"

        msg = sqs.receive_message(QueueUrl=url)["Messages"][0]
        from lead_agent.handlers import aws_lambda

        monkeypatch.setattr(aws_lambda, "get_service", lambda: service)
        event = {
            "Records": [
                {"messageId": msg["MessageId"], "body": msg["Body"]},
                {"messageId": "bad", "body": json.dumps({"run_id": "missing"})},
            ]
        }
        out = aws_lambda.worker_handler(event, None)
        assert out == {"batchItemFailures": [{"itemIdentifier": "bad"}]}
        run_id = json.loads(msg["Body"])["run_id"]
        assert service.get_run(run_id).status == RunStatus.COMPLETED


def test_cli_run_and_list(monkeypatch, tmp_path, capsys):
    from lead_agent import cli, container

    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    container.get_settings.cache_clear()
    original = container.build_service
    monkeypatch.setattr(
        cli,
        "build_service",
        lambda s: original(
            s,
            llm=ScriptedLLM(
                [
                    AssistantMessage("", [call("search_web", query="clínica dental Lima Olivos")]),
                    AssistantMessage(
                        "",
                        [
                            call(
                                "save_lead",
                                name="OrtoDent Los Olivos",
                                sector="Ortodoncia",
                                summary="Consultorio especializado en ortodoncia en Los Olivos.",
                                website="https://ortodent-losolivos.example",
                                location="Los Olivos, Lima",
                                contact={"phones": ["(01) 555-0381"]},
                                source_urls=["https://ortodent-losolivos.example/nosotros"],
                                fit_score=5,
                                fit_reason="Ortodoncia en Lima",
                            )
                        ],
                    ),
                    AssistantMessage("", [call("finish", summary="ok")]),
                ]
            ),
        ),
    )
    assert cli.main(["run", "--sector", "Clínicas dentales", "--location", "Lima", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["leads"][0]["name"] == "OrtoDent Los Olivos"
    assert cli.main(["list"]) == 0
    assert "OrtoDent" in capsys.readouterr().out
    container.get_settings.cache_clear()
