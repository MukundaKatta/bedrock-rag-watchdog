"""Tests for the Datadog push helpers — all offline (stub mode)."""

from ..datadog_push import (
    DatadogConfig,
    MetricPoint,
    create_incident_if_needed,
    push_metrics,
)


def _cfg() -> DatadogConfig:
    return DatadogConfig(api_key="x", app_key="y")


def test_metric_point_defaults_empty_tags():
    m = MetricPoint("rag.drift.embedding_drift", 0.5)
    assert m.name == "rag.drift.embedding_drift"
    assert m.value == 0.5
    assert m.tags == []


def test_metric_point_keeps_provided_tags():
    m = MetricPoint("rag.drift.latency_drift", 0.1, ["env:dev"])
    assert m.tags == ["env:dev"]


def test_push_metrics_stub_counts_all_points():
    metrics = [
        MetricPoint("rag.drift.a", 0.1),
        MetricPoint("rag.drift.b", 0.2),
        MetricPoint("rag.drift.c", 0.3),
    ]
    result = push_metrics(metrics, _cfg(), stub=True)
    assert result["status"] == "stub"
    assert result["count"] == 3


def test_push_metrics_stub_empty_list():
    result = push_metrics([], _cfg(), stub=True)
    assert result["status"] == "stub"
    assert result["count"] == 0


def test_create_incident_returns_none_below_threshold():
    url = create_incident_if_needed(0.05, 0.15, "summary", _cfg(), stub=True)
    assert url is None


def test_create_incident_returns_url_at_threshold():
    url = create_incident_if_needed(0.15, 0.15, "summary", _cfg(), stub=True)
    assert url is not None
    assert "stub" in url


def test_create_incident_returns_url_above_threshold():
    url = create_incident_if_needed(0.9, 0.15, "drift detected", _cfg(), stub=True)
    assert url is not None
    assert url.startswith("https://")
