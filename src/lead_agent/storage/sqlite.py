"""SQLite adapter: zero-infrastructure local storage. Documents stored as JSON + indexed columns."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from lead_agent.domain import Lead, Run
from lead_agent.storage.base import DuplicateLeadError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    lead_id      TEXT PRIMARY KEY,
    query_sector TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_leads_sector ON leads(query_sector, created_at);
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data       TEXT NOT NULL
);
"""


class SQLiteLeadRepository:
    def __init__(self, path: str | Path = "data/leads.db") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may call from a threadpool: one connection guarded by a lock is enough here.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def save_lead(self, lead: Lead) -> None:
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO leads (lead_id, query_sector, run_id, created_at, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        lead.lead_id,
                        lead.query_sector,
                        lead.run_id,
                        lead.created_at.isoformat(),
                        lead.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateLeadError(lead.lead_id) from exc

    def get_lead(self, lead_id: str) -> Lead | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM leads WHERE lead_id = ?", (lead_id,)
            ).fetchone()
        return Lead.model_validate_json(row[0]) if row else None

    def list_leads(self, *, sector: str | None = None, limit: int = 50) -> list[Lead]:
        sql, params = "SELECT data FROM leads", []
        if sector:
            sql += " WHERE query_sector = ?"
            params.append(sector)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Lead.model_validate_json(r[0]) for r in rows]

    def save_run(self, run: Run) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, status, created_at, data) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, data = excluded.data",
                (run.run_id, run.status.value, run.created_at.isoformat(), run.model_dump_json()),
            )

    def list_runs(self, *, limit: int = 20) -> list[Run]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Run.model_validate_json(r[0]) for r in rows]

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row[0]) if row else None
