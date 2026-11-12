"""AWS Lambda controllers.

- api_handler:    API Gateway (HTTP API) -> FastAPI via Mangum. With RUN_MODE=async it only
                  creates the run and enqueues it (202 + run_id).
- worker_handler: SQS -> executes the agent. Reports partial batch failures so only the
                  failed messages are retried (and end up in the DLQ after maxReceiveCount).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from lead_agent.config import get_settings
from lead_agent.container import get_service
from lead_agent.logging_setup import configure_logging

configure_logging(get_settings().log_level)
log = logging.getLogger(__name__)

_api = None


def api_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _api
    if _api is None:
        from mangum import Mangum

        from lead_agent.api.app import app

        _api = Mangum(app, lifespan="off")
    return _api(event, context)


def worker_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    service = get_service()
    failures: list[dict[str, str]] = []
    for record in event.get("Records", []):
        message_id = record.get("messageId", "")
        try:
            run_id = json.loads(record["body"])["run_id"]
            run = service.execute_run(run_id)
            log.info("worker processed run", extra={"run_id": run_id, "status": run.status.value})
        except Exception:
            log.exception("worker failed", extra={"message_id": message_id})
            failures.append({"itemIdentifier": message_id})
    return {"batchItemFailures": failures}
