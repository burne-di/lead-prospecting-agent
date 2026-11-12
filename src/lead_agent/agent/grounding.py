"""Grounding guard: deterministic anti-hallucination check.

Every value the LLM wants to persist (name, emails, phones, website, social profiles, address)
must be traceable to the text of the search results it cites in `source_urls`. Anything that
is not literally present is rejected and the agent is told exactly what to fix.
"""

from __future__ import annotations

import re

from lead_agent.domain.models import LeadDraft, domain_of, normalize_text
from lead_agent.search.base import SearchResult

_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().\-]{5,}\d")
_MIN_PHONE_DIGITS = 7
_ADDRESS_MIN_COVERAGE = 0.7


def _canonical_url(url: str) -> str:
    url = url.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = url.removeprefix("www.")
    return url.rstrip("/")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


class EvidenceStore:
    """Everything the agent has *actually seen* during the run, indexed by URL."""

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def add(self, results: list[SearchResult]) -> None:
        for r in results:
            self._docs[_canonical_url(r.url)] = f"{r.url}\n{r.title}\n{r.content}"

    def has(self, url: str) -> bool:
        return _canonical_url(url) in self._docs

    def text(self, urls: list[str]) -> str:
        return "\n".join(self._docs.get(_canonical_url(u), "") for u in urls)

    def __len__(self) -> int:
        return len(self._docs)


def check_grounding(draft: LeadDraft, evidence: EvidenceStore) -> list[str]:
    """Return a list of human-readable errors (empty list = fully grounded)."""
    errors: list[str] = []

    unknown = [u for u in draft.source_urls if not evidence.has(u)]
    if unknown:
        errors.append(f"source_urls no provienen de search_web: {unknown}")
    cited = [u for u in draft.source_urls if evidence.has(u)]
    if not cited:
        return errors

    raw = evidence.text(cited)
    norm = normalize_text(raw)
    lower = raw.lower()

    if normalize_text(draft.name) not in norm:
        errors.append(f"name '{draft.name}' no aparece literalmente en las fuentes citadas")

    for email in draft.contact.emails:
        if email.lower() not in lower:
            errors.append(f"email '{email}' no aparece en las fuentes citadas: elimínalo")

    source_phones = {_digits(m) for m in _PHONE_CANDIDATE.findall(raw)}
    source_phones = {p for p in source_phones if len(p) >= _MIN_PHONE_DIGITS}
    for phone in draft.contact.phones:
        d = _digits(phone)
        ok = len(d) >= _MIN_PHONE_DIGITS and any(
            p.endswith(d) or d.endswith(p) for p in source_phones
        )
        if not ok:
            errors.append(f"teléfono '{phone}' no aparece en las fuentes citadas: elimínalo")

    if draft.website:
        domain = domain_of(draft.website)
        if not domain or domain not in lower:
            errors.append(f"website '{draft.website}' no aparece en las fuentes citadas")

    for profile in draft.contact.social_profiles:
        if _canonical_url(profile) not in lower:
            errors.append(f"perfil social '{profile}' no aparece en las fuentes citadas")

    if draft.contact.address:
        tokens = [t for t in re.findall(r"[a-z0-9]+", normalize_text(draft.contact.address))]
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
        if tokens:
            coverage = sum(1 for t in tokens if t in norm) / len(tokens)
            numbers_ok = all(t in norm for t in tokens if t.isdigit())
            if coverage < _ADDRESS_MIN_COVERAGE or not numbers_ok:
                errors.append(
                    f"dirección '{draft.contact.address}' no coincide con la fuente: "
                    "cópiala literalmente o déjala en null"
                )

    return errors
