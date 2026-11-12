"""How a run gets executed: inline (local/sync) or through SQS (production/async)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from lead_agent.domain import Run


class Dispatcher(Protocol):
    is_async: bool

    def dispatch(self, run: Run, execute: Callable[[str], Run]) -> Run: ...


class InlineDispatcher:
    is_async = False

    def dispatch(self, run: Run, execute: Callable[[str], Run]) -> Run:
        return execute(run.run_id)


class SQSDispatcher:
    is_async = True

    def __init__(self, queue_url: str, *, region: str | None = None, client: Any = None) -> None:
        import boto3

        self._queue_url = queue_url
        self._sqs = client or boto3.client("sqs", region_name=region)

    def dispatch(self, run: Run, execute: Callable[[str], Run]) -> Run:
        self._sqs.send_message(
            QueueUrl=self._queue_url, MessageBody=json.dumps({"run_id": run.run_id})
        )
        return run
