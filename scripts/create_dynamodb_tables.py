"""Create the DynamoDB tables (DynamoDB Local or AWS) using the settings from .env."""

import boto3

from lead_agent.config import get_settings
from lead_agent.storage.dynamodb import create_tables


def main() -> None:
    s = get_settings()
    resource = boto3.resource(
        "dynamodb",
        region_name=s.aws_region,
        endpoint_url=s.dynamodb_endpoint_url,
        # DynamoDB Local accepts any credentials
        **(
            {"aws_access_key_id": "local", "aws_secret_access_key": "local"}
            if s.dynamodb_endpoint_url
            else {}
        ),
    )
    create_tables(resource, leads_table=s.dynamodb_leads_table, runs_table=s.dynamodb_runs_table)
    print("Tablas:", [t.name for t in resource.tables.all()])


if __name__ == "__main__":
    main()
