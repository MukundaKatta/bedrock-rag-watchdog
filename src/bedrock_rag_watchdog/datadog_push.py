"""
Datadog metric + incident push helpers.

In production these call the Datadog API. In tests pass `stub=True` to
skip all network calls (the same stub pattern used throughout birddog,
ragvitals, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DatadogConfig:
    api_key: str
    app_key: str
    site: str = "datadoghq.com"
    metric_prefix: str = "rag.drift"


class MetricPoint:
    def __init__(self, name: str, value: float, tags: List[str] | None = None):
        self.name = name
        self.value = value
        self.tags = tags or []


def push_metrics(
    metrics: List[MetricPoint],
    cfg: DatadogConfig,
    stub: bool = False,
) -> Dict[str, Any]:
    """Push custom metrics to Datadog.

    In stub mode: log and return without any HTTP call.
    In production: calls the Datadog metrics API.
    """
    if stub:
        for m in metrics:
            print(f"[stub] DD metric {m.name}={m.value} tags={m.tags}")
        return {"status": "stub", "count": len(metrics)}

    try:
        from datadog_api_client import ApiClient, Configuration
        from datadog_api_client.v2.api.metrics_api import MetricsApi
        from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
        from datadog_api_client.v2.model.metric_payload import MetricPayload
        from datadog_api_client.v2.model.metric_point import MetricPoint as DDMetricPoint
        from datadog_api_client.v2.model.metric_series import MetricSeries
        import time
    except ImportError as e:
        raise RuntimeError(f"datadog-api-client not installed: {e}") from e

    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = cfg.api_key
    configuration.api_key["appKeyAuth"] = cfg.app_key
    configuration.server_variables["site"] = cfg.site

    series = [
        MetricSeries(
            metric=m.name,
            type=MetricIntakeType.GAUGE,
            points=[DDMetricPoint(timestamp=int(time.time()), value=m.value)],
            tags=m.tags,
        )
        for m in metrics
    ]

    with ApiClient(configuration) as api_client:
        api_instance = MetricsApi(api_client)
        response = api_instance.submit_metrics(body=MetricPayload(series=series))
        return {"status": "ok", "count": len(series), "errors": response.errors}


def create_incident_if_needed(
    drift_max: float,
    threshold: float,
    summary: str,
    cfg: DatadogConfig,
    stub: bool = False,
) -> Optional[str]:
    """Create a Datadog incident when drift exceeds threshold.

    Returns the incident URL (or a stub indicator), or None if no incident needed.
    """
    if drift_max < threshold:
        return None

    if stub:
        print(f"[stub] DD incident: {summary}")
        return "https://app.datadoghq.com/incidents/stub-00001"

    try:
        from datadog_api_client import ApiClient, Configuration
        from datadog_api_client.v2.api.incidents_api import IncidentsApi
        from datadog_api_client.v2.model.incident_create_attributes import IncidentCreateAttributes
        from datadog_api_client.v2.model.incident_create_data import IncidentCreateData
        from datadog_api_client.v2.model.incident_create_request import IncidentCreateRequest
        from datadog_api_client.v2.model.incident_type import IncidentType
    except ImportError as e:
        raise RuntimeError(f"datadog-api-client not installed: {e}") from e

    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = cfg.api_key
    configuration.api_key["appKeyAuth"] = cfg.app_key
    configuration.server_variables["site"] = cfg.site
    configuration.unstable_operations["create_incident"] = True

    body = IncidentCreateRequest(
        data=IncidentCreateData(
            type=IncidentType.INCIDENTS,
            attributes=IncidentCreateAttributes(
                title=f"RAG Drift Alert: max={drift_max:.3f} (threshold={threshold})",
                customer_impacted=False,
                fields={},
                notification_handles=[],
                initial_timeline_cells=[
                    {"cellType": "markdown", "content": {"content": summary}},
                ],
            ),
        )
    )
    with ApiClient(configuration) as api_client:
        api_instance = IncidentsApi(api_client)
        incident = api_instance.create_incident(body)
        return f"https://app.datadoghq.com/incidents/{incident.data.id}"
