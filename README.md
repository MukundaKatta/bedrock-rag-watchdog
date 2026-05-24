# bedrock-rag-watchdog

RAG drift monitor for AWS Bedrock AgentCore + Datadog. Built for the AWS + Datadog Generative AI Hackathon 2026.

## What it does

Runs one monitoring cycle every N minutes:

1. Pulls recent retrieval traces (from S3 or provided inline).
2. Computes five drift dimensions: `embedding_drift`, `retrieval_drift`, `response_drift`, `latency_drift`, `coverage_drift`.
3. Pushes the five metrics to Datadog as `rag.drift.*` gauges.
4. If any dimension exceeds threshold, asks Claude on Bedrock to write a one-paragraph human-readable incident summary, then creates a Datadog incident.

## Quick start (stub mode — no AWS or Datadog needed)

```bash
pip install bedrock-rag-watchdog
bedrock-rag-watchdog
```

## Production

```bash
export DD_API_KEY=...
export DD_APP_KEY=...
export AWS_REGION=us-east-1
export S3_TRACE_BUCKET=my-rag-traces
export STUB=0
bedrock-rag-watchdog
```

## Drift dimensions

| Dimension | Measures |
|---|---|
| `embedding_drift` | Cosine distance between query and retrieved vectors |
| `retrieval_drift` | NDCG@5 change relative to baseline |
| `response_drift` | Jaccard divergence between response text batches |
| `latency_drift` | p95 latency ratio to baseline |
| `coverage_drift` | Unique doc coverage ratio change |

## Tests

```bash
pytest src/  # 17 tests, all offline
```
