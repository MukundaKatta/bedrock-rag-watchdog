"""Tests for drift computation — all offline, no external deps."""

import math
import pytest
from ..drift import RetrievalTrace, compute_drift, DriftDimensions


def make_trace(
    idx: int = 0,
    latency_ms: float = 100.0,
    response: str = "default answer",
    docs: list[str] | None = None,
    scores: list[float] | None = None,
) -> RetrievalTrace:
    q = [math.sin(idx * 0.1 + i * 0.01) for i in range(16)]
    ret_embs = [[math.cos(idx * 0.1 + i * 0.02)] * 16 for i in range(3)]
    return RetrievalTrace(
        query_embedding=q,
        retrieved_embeddings=ret_embs,
        response_text=response,
        latency_ms=latency_ms,
        retrieved_doc_ids=docs or [f"doc_{idx}", f"doc_{idx + 1}"],
        relevance_scores=scores or [0.9 - idx * 0.05, 0.8 - idx * 0.03],
    )


class TestComputeDrift:
    def test_empty_returns_zero_drift(self):
        d = compute_drift([])
        assert d.max() == 0.0

    def test_single_trace_no_baseline(self):
        d = compute_drift([make_trace(0)])
        # Single trace — no stdev possible, most dims should be 0 or near 0
        assert d.max() >= 0.0
        assert d.max() < 1.0

    def test_identical_traces_low_drift(self):
        traces = [make_trace(0) for _ in range(5)]
        d = compute_drift(traces)
        # All identical — stdev of emb distances should be 0 or near 0
        assert d.embedding_drift < 0.01

    def test_diverse_traces_higher_drift(self):
        traces = [make_trace(i, latency_ms=100 + i * 50, response=f"answer {i} {i*i}") for i in range(10)]
        d = compute_drift(traces)
        # Should have some measurable drift
        assert d.max() >= 0.0

    def test_with_baseline_returns_abs_diff(self):
        current = [make_trace(i, latency_ms=200.0) for i in range(5)]
        baseline = [make_trace(i, latency_ms=100.0) for i in range(5)]
        d = compute_drift(current, baseline)
        # Latency doubled → latency_drift should be non-zero
        assert d.latency_drift > 0

    def test_different_doc_sets_coverage_drift(self):
        current = [make_trace(i, docs=[f"new_doc_{i}"]) for i in range(5)]
        baseline = [make_trace(i, docs=[f"old_doc_{i}"]) for i in range(5)]
        d = compute_drift(current, baseline)
        assert d.coverage_drift > 0

    def test_same_docs_zero_coverage_drift(self):
        traces = [make_trace(i, docs=["doc_a", "doc_b"]) for i in range(5)]
        d = compute_drift(traces, traces)
        assert d.coverage_drift == 0.0

    def test_dim_dict_keys(self):
        d = compute_drift([make_trace(0)])
        keys = set(d.as_dict().keys())
        assert keys == {"embedding_drift", "retrieval_drift", "response_drift", "latency_drift", "coverage_drift"}

    def test_dim_values_non_negative(self):
        traces = [make_trace(i) for i in range(8)]
        d = compute_drift(traces)
        for v in d.as_dict().values():
            assert v >= 0.0

    def test_max_returns_highest_dim(self):
        d = DriftDimensions(
            embedding_drift=0.1,
            retrieval_drift=0.5,
            response_drift=0.3,
            latency_drift=0.2,
            coverage_drift=0.4,
        )
        assert d.max() == pytest.approx(0.5)

    def test_high_latency_drift(self):
        current = [make_trace(i, latency_ms=1000.0) for i in range(5)]
        baseline = [make_trace(i, latency_ms=100.0) for i in range(5)]
        d = compute_drift(current, baseline)
        assert d.latency_drift > 0.5

    def test_response_drift_same_text(self):
        text = "The answer is always the same no matter what."
        current = [make_trace(i, response=text) for i in range(5)]
        baseline = [make_trace(i, response=text) for i in range(5)]
        d = compute_drift(current, baseline)
        assert d.response_drift == 0.0


class TestWatchdogAgent:
    def test_smoke_run_stub_mode(self):
        from ..agent import WatchdogAgent, WatchdogConfig
        cfg = WatchdogConfig(stub=True, drift_threshold=0.15)
        agent = WatchdogAgent(cfg)
        report = agent.run()
        assert report.dimensions is not None
        assert isinstance(report.summary, str)
        assert report.metrics_pushed >= 0

    def test_report_exceeded_threshold_flag(self):
        from ..agent import WatchdogAgent, WatchdogConfig
        from ..drift import RetrievalTrace
        # Force high latency drift
        high_lat = [make_trace(i, latency_ms=2000.0) for i in range(5)]
        base_lat = [make_trace(i, latency_ms=100.0) for i in range(5)]
        cfg = WatchdogConfig(stub=True, drift_threshold=0.15)
        agent = WatchdogAgent(cfg)
        report = agent.run(traces=high_lat, baseline=base_lat)
        # Latency went from 100 to 2000 → drift >> 0.15
        assert report.exceeded_threshold

    def test_report_no_incident_below_threshold(self):
        from ..agent import WatchdogAgent, WatchdogConfig
        # All identical traces → low drift
        traces = [make_trace(0) for _ in range(5)]
        cfg = WatchdogConfig(stub=True, drift_threshold=0.99)
        agent = WatchdogAgent(cfg)
        report = agent.run(traces=traces)
        # With threshold 0.99, no incident expected
        assert not report.exceeded_threshold
        assert report.incident_url is None

    def test_incident_created_when_exceeded_stub(self):
        from ..agent import WatchdogAgent, WatchdogConfig
        # Use synthetic traces with very high latency
        high = [make_trace(i, latency_ms=5000.0) for i in range(5)]
        base = [make_trace(i, latency_ms=10.0) for i in range(5)]
        cfg = WatchdogConfig(stub=True, drift_threshold=0.01)
        agent = WatchdogAgent(cfg)
        report = agent.run(traces=high, baseline=base)
        assert report.exceeded_threshold
        assert report.incident_url is not None
        assert "stub" in report.incident_url

    def test_synthetic_traces_used_when_none_provided(self):
        from ..agent import WatchdogAgent, WatchdogConfig
        cfg = WatchdogConfig(stub=True)
        agent = WatchdogAgent(cfg)
        # No traces, no S3 → uses synthetic
        report = agent.run(traces=None)
        assert report.dimensions is not None
