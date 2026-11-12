"""Both repositories satisfy the same contract (SQLite and DynamoDB via moto)."""

import boto3
import pytest
from moto import mock_aws

from conftest import lead_args
from lead_agent.domain import Lead, LeadDraft, Run, RunRequest, RunStatus
from lead_agent.storage.base import DuplicateLeadError
from lead_agent.storage.dynamodb import DynamoDBLeadRepository, create_tables
from lead_agent.storage.sqlite import SQLiteLeadRepository


@pytest.fixture(params=["sqlite", "dynamodb"])
def repository(request, monkeypatch):
    if request.param == "sqlite":
        yield SQLiteLeadRepository(":memory:")
        return
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        res = boto3.resource("dynamodb", region_name="us-east-1")
        create_tables(res, leads_table="leads", runs_table="runs")
        yield DynamoDBLeadRepository(leads_table="leads", runs_table="runs", resource=res)


REQ = RunRequest(sector="Clínicas dentales", location="Lima, Perú")


def _lead(**kw) -> Lead:
    return Lead.from_draft(LeadDraft.model_validate(lead_args(**kw)), run_id="r1", request=REQ)


def test_save_and_get_lead(repository):
    lead = _lead()
    repository.save_lead(lead)
    got = repository.get_lead(lead.lead_id)
    assert got == lead


def test_conditional_insert_rejects_duplicates(repository):
    repository.save_lead(_lead())
    with pytest.raises(DuplicateLeadError):
        repository.save_lead(_lead(name="Acme Dental"))


def test_list_by_sector(repository):
    repository.save_lead(_lead())
    repository.save_lead(_lead(website="https://otro.example", name="Otro"))
    other = Lead.from_draft(
        LeadDraft.model_validate(lead_args(website="https://x.example")),
        run_id="r2",
        request=RunRequest(sector="Agencias de marketing", location="Lima"),
    )
    repository.save_lead(other)
    assert len(repository.list_leads(sector="Clínicas dentales")) == 2
    assert len(repository.list_leads()) == 3


def test_run_upsert_roundtrip(repository):
    run = Run(request=REQ)
    repository.save_run(run)
    run.status, run.steps = RunStatus.COMPLETED, 4
    run.usage.input_tokens = 1234
    repository.save_run(run)
    got = repository.get_run(run.run_id)
    assert got.status == RunStatus.COMPLETED and got.steps == 4 and got.usage.input_tokens == 1234
    assert repository.get_run("nope") is None


def test_list_runs_most_recent_first(repository):
    from datetime import timedelta

    old = Run(request=REQ)
    new = Run(request=REQ, created_at=old.created_at + timedelta(seconds=5))
    repository.save_run(old)
    repository.save_run(new)
    assert [r.run_id for r in repository.list_runs()] == [new.run_id, old.run_id]
