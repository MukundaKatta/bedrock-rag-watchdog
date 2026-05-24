"""
Bedrock AgentCore orchestration layer.

WatchdogAgent runs one monitoring cycle:
  1. Pull recent retrieval traces from S3 (or use provided traces).
  2. Compute RAG drift across five dimensions.
  3. Push the five metrics to Datadog.
  4. If any dimension exceeds threshold, ask Claude on Bedrock to write a
     one-paragraph human-readable summary, then create a Datadog incident.

When stub=True (default), no AWS or Datadog calls are made — safe to run
in CI, demos, and offline smoke tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .drift import DriftDimensions, RetrievalTrace, compute_drift
from .datadog_push import DatadogConfig, MetricPoint, create_incident_if_needed, push_metrics


@dataclass
class WatchdogConfig:
    # Datadog
    datadog_api_key: str = ""
    datadog_app_key: str = ""
    datadog_site: str = "datadoghq.com"
    metric_prefix: str = "rag.drift"
    drift_threshold: float = 0.15  # alert when any dimension exceeds this

    # Bedrock
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

    # S3 trace store (optional)
    s3_bucket: str = ""
    s3_prefix: str = "rag-traces/"

    # Tags applied to all metrics
    extra_tags: List[str] = field(default_factory=list)

    # Stub mode — skip all external calls
    stub: bool = True


@dataclass
class DriftReport:
    dimensions: DriftDimensions
    metrics_pushed: int
    incident_url: Optional[str]
    summary: str
    exceeded_threshold: bool


class WatchdogAgent:
    def __init__(self, cfg: WatchdogConfig):
        self.cfg = cfg

    def _load_traces_from_s3(self) -> List[RetrievalTrace]:
        """Pull JSON trace records from S3. Returns empty list in stub mode."""
        if self.cfg.stub or not self.cfg.s3_bucket:
            return []
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self.cfg.bedrock_region)
            objects = s3.list_objects_v2(
                Bucket=self.cfg.s3_bucket, Prefix=self.cfg.s3_prefix
            ).get("Contents", [])
            traces = []
            for obj in sorted(objects, key=lambda o: o["LastModified"])[-100:]:
                body = s3.get_object(Bucket=self.cfg.s3_bucket, Key=obj["Key"])["Body"].read()
                data = json.loads(body)
                traces.append(RetrievalTrace(**data))
            return traces
        except Exception as exc:
            print(f"[warn] S3 load failed: {exc}")
            return []

    def _build_summary(self, dims: DriftDimensions) -> str:
        """Ask Claude on Bedrock to write a drift summary (stub: template)."""
        exceeded = [k for k, v in dims.as_dict().items() if v >= self.cfg.drift_threshold]
        if not exceeded:
            return "RAG pipeline is operating within normal drift bounds."

        template = (
            f"RAG drift alert: {', '.join(exceeded)} exceeded threshold "
            f"{self.cfg.drift_threshold}. "
            f"Max dimension: {dims.max():.3f}. "
            f"Investigate recent embedding model updates or corpus changes."
        )

        if self.cfg.stub:
            return template

        try:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=self.cfg.bedrock_region)
            prompt = (
                f"You are an on-call SRE. Summarize this RAG drift report in one paragraph "
                f"for a non-technical stakeholder. Dimensions exceeded: {exceeded}. "
                f"Values: {dims.as_dict()}. Threshold: {self.cfg.drift_threshold}."
            )
            response = bedrock.invoke_model(
                modelId=self.cfg.bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            body = json.loads(response["body"].read())
            return body["content"][0]["text"]
        except Exception as exc:
            print(f"[warn] Bedrock invoke failed: {exc}")
            return template

    def run(
        self,
        traces: Optional[List[RetrievalTrace]] = None,
        baseline: Optional[List[RetrievalTrace]] = None,
    ) -> DriftReport:
        """
        Run one monitoring cycle.

        Args:
            traces:   List of recent RetrievalTrace objects. If None, loaded from S3.
            baseline: Historical baseline traces for comparison. If None, drift is
                      computed as intra-batch variance.

        Returns:
            DriftReport with all computed values.
        """
        if traces is None:
            traces = self._load_traces_from_s3()

        if not traces:
            # Generate synthetic traces for demo/smoke-test
            import math
            traces = [
                RetrievalTrace(
                    query_embedding=[math.sin(i * 0.1)] * 32,
                    retrieved_embeddings=[[math.cos(i * 0.1 + j * 0.05)] * 32 for j in range(3)],
                    response_text=f"Answer {i}: The retrieved context shows relevant information about the query.",
                    latency_ms=120.0 + i * 5,
                    retrieved_doc_ids=[f"doc_{i}", f"doc_{i+1}", f"doc_{i+2}"],
                    relevance_scores=[0.9 - i * 0.05, 0.8 - i * 0.03, 0.7],
                )
                for i in range(10)
            ]
            print(f"[info] Using {len(traces)} synthetic traces (no S3 data).")

        # Compute drift
        dims = compute_drift(traces, baseline)
        exceeded = dims.max() >= self.cfg.drift_threshold

        # Build summary
        summary = self._build_summary(dims)

        # Push metrics to Datadog
        dd_cfg = DatadogConfig(
            api_key=self.cfg.datadog_api_key,
            app_key=self.cfg.datadog_app_key,
            site=self.cfg.datadog_site,
            metric_prefix=self.cfg.metric_prefix,
        )
        tags = [f"env:{os.environ.get('ENVIRONMENT', 'dev')}"] + self.cfg.extra_tags
        metrics = [
            MetricPoint(f"{self.cfg.metric_prefix}.{k}", v, tags)
            for k, v in dims.as_dict().items()
        ]
        push_result = push_metrics(metrics, dd_cfg, stub=self.cfg.stub)
        pushed = push_result.get("count", 0)

        # Create incident if threshold exceeded
        incident_url = create_incident_if_needed(
            dims.max(), self.cfg.drift_threshold, summary, dd_cfg, stub=self.cfg.stub
        )

        return DriftReport(
            dimensions=dims,
            metrics_pushed=pushed,
            incident_url=incident_url,
            summary=summary,
            exceeded_threshold=exceeded,
        )
