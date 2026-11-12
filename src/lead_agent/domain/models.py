"""Domain entities. Pure Pydantic: no dependency on LLMs, search providers or databases."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SOCIAL_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "wa.me",
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_text(value: str) -> str:
    """Lowercase, strip accents and collapse whitespace (for tolerant literal matching)."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip().lower()


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return host or None


def is_social_url(url: str) -> bool:
    host = domain_of(url) or ""
    return any(host == d or host.endswith("." + d) for d in _SOCIAL_DOMAINS)


# --------------------------------------------------------------------------- lead


class Contact(BaseModel):
    """Datos de contacto copiados literalmente de la fuente. Listas vacías si no hay."""

    model_config = ConfigDict(extra="forbid")

    emails: list[str] = Field(
        default_factory=list, description="Emails tal como aparecen en la fuente. Nunca inventados."
    )
    phones: list[str] = Field(
        default_factory=list, description="Teléfonos tal como aparecen en la fuente."
    )
    address: str | None = Field(
        default=None, description="Dirección literal de la fuente, o null si no aparece."
    )
    social_profiles: list[str] = Field(
        default_factory=list, description="URLs de Facebook, Instagram, LinkedIn, etc."
    )

    @field_validator("emails")
    @classmethod
    def _valid_emails(cls, v: list[str]) -> list[str]:
        out = []
        for e in v:
            e = e.strip()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e):
                raise ValueError(f"email con formato inválido: {e!r}")
            out.append(e)
        return list(dict.fromkeys(out))

    @field_validator("phones", "social_profiles")
    @classmethod
    def _strip_dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(x.strip() for x in v if x.strip()))


class LeadDraft(BaseModel):
    """Contrato de la tool save_lead: lo que el LLM extrae de la evidencia."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=200, description="Nombre literal en la fuente.")
    sector: str = Field(min_length=2, max_length=120, description="Sector pedido o más específico.")
    summary: str = Field(
        min_length=10,
        max_length=600,
        description="1-2 frases en español con SOLO lo que la fuente dice que hacen.",
    )
    website: str | None = Field(
        default=None, description="Sitio propio del negocio (no redes sociales), o null."
    )
    location: str | None = Field(default=None, description="Distrito/ciudad según la fuente.")
    contact: Contact = Field(default_factory=Contact)
    source_urls: list[str] = Field(
        min_length=1, description="URLs de resultados de search_web que respaldan los datos."
    )
    fit_score: int = Field(ge=1, le=5, description="Rúbrica 1-5 del system prompt.")
    fit_reason: str = Field(min_length=5, max_length=300, description="Justificación breve.")

    @field_validator("website")
    @classmethod
    def _website_not_social(cls, v: str | None) -> str | None:
        if v and is_social_url(v):
            raise ValueError("un perfil de red social va en contact.social_profiles, no en website")
        return v or None


def compute_lead_id(draft: LeadDraft) -> str:
    """Stable id: website domain if any, otherwise normalized name + location."""
    domain = domain_of(draft.website)
    key = (
        f"web:{domain}"
        if domain
        else f"name:{normalize_text(draft.name)}|{normalize_text(draft.location or '')}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class RunRequest(BaseModel):
    """Parámetros de entrada configurables del agente."""

    model_config = ConfigDict(str_strip_whitespace=True)

    sector: str = Field(min_length=2, max_length=120, examples=["Clínicas dentales"])
    location: str = Field(min_length=2, max_length=120, examples=["Lima, Perú"])
    max_leads: int = Field(default=5, ge=1, le=20)


class Lead(LeadDraft):
    lead_id: str
    run_id: str
    query_sector: str
    query_location: str
    created_at: datetime = Field(default_factory=utcnow)

    @classmethod
    def from_draft(cls, draft: LeadDraft, *, run_id: str, request: RunRequest) -> Lead:
        return cls(
            **draft.model_dump(),
            lead_id=compute_lead_id(draft),
            run_id=run_id,
            query_sector=request.sector,
            query_location=request.location,
        )


# --------------------------------------------------------------------------- run


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {RunStatus.COMPLETED, RunStatus.MAX_STEPS, RunStatus.FAILED}


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class TraceCall(BaseModel):
    """One tool call made by the LLM and what the system answered."""

    tool: str
    input: str  # compact, truncated JSON of the arguments
    status: str  # ok | saved | rejected | duplicate | error
    detail: str = ""


class TraceStep(BaseModel):
    """One reasoning step of the ReAct loop (auditable evidence of the agent's decisions)."""

    step: int
    thought: str = ""
    calls: list[TraceCall] = Field(default_factory=list)


class Run(BaseModel):
    """Una ejecución del agente: auditoría de qué hizo y cuánto costó."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    request: RunRequest
    status: RunStatus = RunStatus.QUEUED
    steps: int = 0
    searches: int = 0
    rejected_saves: int = 0
    duplicates: int = 0
    rejections: list[str] = Field(default_factory=list)  # why save_lead was rejected (audit)
    trace: list[TraceStep] = Field(default_factory=list)
    lead_ids: list[str] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    summary: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
