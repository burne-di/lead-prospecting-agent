from conftest import NOISE, SOURCE, SRC_URL, lead_args
from lead_agent.agent.grounding import EvidenceStore, check_grounding
from lead_agent.domain import LeadDraft


def _check(**overrides) -> list[str]:
    ev = EvidenceStore()
    ev.add([SOURCE, NOISE])
    return check_grounding(LeadDraft.model_validate(lead_args(**overrides)), ev)


def test_fully_grounded_lead_passes():
    assert _check() == []


def test_hallucinated_email_is_rejected():
    errors = _check(contact=dict(emails=["info@acme-dental.example"]))
    assert len(errors) == 1 and "info@acme-dental.example" in errors[0]


def test_phone_matches_by_digits_regardless_of_format():
    assert _check(contact=dict(phones=["+51 (955) 000-111"])) == []


def test_invented_phone_is_rejected():
    assert _check(contact=dict(phones=["955 000 999"]))


def test_unknown_source_url_is_rejected():
    errors = _check(source_urls=["https://never-seen.example"])
    assert any("no provienen de search_web" in e for e in errors)


def test_contact_from_other_source_is_rejected():
    # the directory's sales email exists in the evidence, but not in the cited source
    assert _check(contact=dict(emails=["ventas@directorio.example"]))


def test_name_must_appear_in_source_accent_insensitive():
    assert _check(name="ACME DENTAL") == []
    assert _check(name="Acme Dental Group SAC")


def test_website_domain_must_appear():
    assert _check(website="https://acme-dental-lima.example")


def test_address_requires_numbers_to_match():
    assert _check(contact=dict(address="Av. Pardo 999, Miraflores"))


def test_evidence_url_canonicalization():
    ev = EvidenceStore()
    ev.add([SOURCE])
    assert ev.has("http://www.ACME-dental.example/contacto/")
    assert not ev.has(SRC_URL + "/otra")
