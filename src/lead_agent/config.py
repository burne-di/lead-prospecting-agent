"""Typed settings loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/lead_agent/config.py -> repo root


class Settings(BaseSettings):
    # The project's .env is found from any working directory (later files win: ./.env overrides).
    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["anthropic", "bedrock", "openai"] = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    aws_region: str = "us-east-1"

    # Search
    search_provider: Literal["mock", "tavily"] = "mock"
    tavily_api_key: SecretStr | None = None
    search_max_results: int = 6

    # Storage
    storage_backend: Literal["sqlite", "dynamodb"] = "sqlite"
    sqlite_path: str = "data/leads.db"
    dynamodb_leads_table: str = "leads"
    dynamodb_runs_table: str = "lead_runs"
    dynamodb_endpoint_url: str | None = None

    # Agent
    agent_max_steps: int = 15
    agent_max_searches: int = 5

    # Execution: "sync" runs the agent inside the request; "async" enqueues to SQS
    run_mode: Literal["sync", "async"] = "sync"
    runs_queue_url: str | None = None

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"  # Lambda always logs JSON

    @field_validator("*", mode="before")
    @classmethod
    def _empty_as_none(cls, v: object) -> object:
        # `KEY=` in .env means "not set", not an empty string.
        return None if isinstance(v, str) and v.strip() == "" else v


@lru_cache
def get_settings() -> Settings:
    return Settings()


def secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None
