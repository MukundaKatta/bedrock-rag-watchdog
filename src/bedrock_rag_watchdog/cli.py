"""CLI entry point for bedrock-rag-watchdog."""

from __future__ import annotations

import os
import json
from .agent import WatchdogAgent, WatchdogConfig


def main() -> None:
    cfg = WatchdogConfig(
        datadog_api_key=os.environ.get("DD_API_KEY", ""),
        datadog_app_key=os.environ.get("DD_APP_KEY", ""),
        datadog_site=os.environ.get("DD_SITE", "datadoghq.com"),
        bedrock_region=os.environ.get("AWS_REGION", "us-east-1"),
        s3_bucket=os.environ.get("S3_TRACE_BUCKET", ""),
        drift_threshold=float(os.environ.get("DRIFT_THRESHOLD", "0.15")),
        stub=os.environ.get("STUB", "1") == "1",
    )

    print("bedrock-rag-watchdog")
    print(f"  stub mode: {cfg.stub}")
    print(f"  threshold: {cfg.drift_threshold}")
    print()

    agent = WatchdogAgent(cfg)
    report = agent.run()

    print(f"Drift dimensions:")
    for k, v in report.dimensions.as_dict().items():
        flag = " <-- ALERT" if v >= cfg.drift_threshold else ""
        print(f"  {k:20s}: {v:.4f}{flag}")

    print(f"\nMetrics pushed: {report.metrics_pushed}")
    print(f"Summary: {report.summary}")

    if report.incident_url:
        print(f"Incident created: {report.incident_url}")
    else:
        print("No incident (all dimensions within threshold).")

    print(f"\nFull report:\n{json.dumps({'exceeded': report.exceeded_threshold, 'dims': report.dimensions.as_dict(), 'incident': report.incident_url}, indent=2)}")


if __name__ == "__main__":
    main()
