"""DynamoDB adapter (AWS or DynamoDB Local). Conditional writes give atomic de-duplication."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lead_agent.domain import Lead, Run
from lead_agent.storage.base import DuplicateLeadError

SECTOR_INDEX = "sector-index"
RUN_TTL = timedelta(days=90)


def _to_item(model_json: str) -> dict[str, Any]:
    # Store plain JSON types; drop None/empty values (DynamoDB rejects empty strings in keys).
    return {k: v for k, v in json.loads(model_json).items() if v is not None}


class DynamoDBLeadRepository:
    def __init__(
        self,
        *,
        leads_table: str,
        runs_table: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        resource: Any | None = None,
    ) -> None:
        dynamodb = resource or boto3.resource(
            "dynamodb", region_name=region, endpoint_url=endpoint_url
        )
        self._leads = dynamodb.Table(leads_table)
        self._runs = dynamodb.Table(runs_table)

    def save_lead(self, lead: Lead) -> None:
        try:
            self._leads.put_item(
                Item=_to_item(lead.model_dump_json()),
                ConditionExpression="attribute_not_exists(lead_id)",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise DuplicateLeadError(lead.lead_id) from exc
            raise

    def get_lead(self, lead_id: str) -> Lead | None:
        item = self._leads.get_item(Key={"lead_id": lead_id}).get("Item")
        return Lead.model_validate(item) if item else None

    def list_leads(self, *, sector: str | None = None, limit: int = 50) -> list[Lead]:
        if sector:
            from boto3.dynamodb.conditions import Key

            resp = self._leads.query(
                IndexName=SECTOR_INDEX,
                KeyConditionExpression=Key("query_sector").eq(sector),
                ScanIndexForward=False,
                Limit=limit,
            )
            items = resp.get("Items", [])
        else:
            items = self._leads.scan(Limit=limit).get("Items", [])  # PoC only; avoid scans in prod
            items.sort(key=lambda i: i.get("created_at", ""), reverse=True)
        return [Lead.model_validate(i) for i in items[:limit]]

    def save_run(self, run: Run) -> None:
        item = _to_item(run.model_dump_json())
        item["expires_at"] = int((run.created_at + RUN_TTL).timestamp())
        self._runs.put_item(Item=item)

    def list_runs(self, *, limit: int = 20) -> list[Run]:
        # PoC: scan. In production, a GSI (e.g. tenant_id + created_at) would serve this query.
        items = self._runs.scan().get("Items", [])
        items.sort(key=lambda i: i.get("created_at", ""), reverse=True)
        runs = []
        for item in items[:limit]:
            item.pop("expires_at", None)
            runs.append(Run.model_validate(item))
        return runs

    def get_run(self, run_id: str) -> Run | None:
        item = self._runs.get_item(Key={"run_id": run_id}).get("Item")
        if not item:
            return None
        item.pop("expires_at", None)
        return Run.model_validate(item)


def create_tables(resource: Any, *, leads_table: str, runs_table: str) -> None:
    """Same key schema as infra/template.yaml (used by DynamoDB Local and tests)."""
    existing = {t.name for t in resource.tables.all()}
    if leads_table not in existing:
        resource.create_table(
            TableName=leads_table,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "lead_id", "AttributeType": "S"},
                {"AttributeName": "query_sector", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "lead_id", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": SECTOR_INDEX,
                    "KeySchema": [
                        {"AttributeName": "query_sector", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        )
    if runs_table not in existing:
        resource.create_table(
            TableName=runs_table,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
        )
