import pytest
from pydantic import ValidationError

from conftest import lead_args
from lead_agent.domain import Lead, LeadDraft, RunRequest
from lead_agent.domain.models import compute_lead_id


def test_social_profile_as_website_is_rejected():
    with pytest.raises(ValidationError, match="social_profiles"):
        LeadDraft.model_validate(lead_args(website="https://facebook.com/acme"))


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        LeadDraft.model_validate(lead_args(ceo="Juan"))


def test_fit_score_range():
    with pytest.raises(ValidationError):
        LeadDraft.model_validate(lead_args(fit_score=7))


def test_lead_id_is_stable_by_domain():
    a = LeadDraft.model_validate(lead_args())
    b = LeadDraft.model_validate(lead_args(name="Acme", website="http://www.acme-dental.example/"))
    assert compute_lead_id(a) == compute_lead_id(b)


def test_lead_id_falls_back_to_name_and_location():
    a = LeadDraft.model_validate(lead_args(website=None, name="Clínica Ñandú"))
    b = LeadDraft.model_validate(lead_args(website=None, name="clinica ñandu"))
    assert compute_lead_id(a) == compute_lead_id(b)


def test_lead_from_draft_keeps_query():
    req = RunRequest(sector="Clínicas dentales", location="Lima, Perú")
    lead = Lead.from_draft(LeadDraft.model_validate(lead_args()), run_id="r1", request=req)
    assert lead.query_location == "Lima, Perú" and lead.run_id == "r1"


def test_run_request_limits():
    with pytest.raises(ValidationError):
        RunRequest(sector="x", location="Lima")
    with pytest.raises(ValidationError):
        RunRequest(sector="Dentistas", location="Lima", max_leads=100)
