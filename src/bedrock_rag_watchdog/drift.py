"""
RAG drift detection — standalone, no ragdrift-mcp dependency.

Computes five drift dimensions from a batch of retrieval traces:
  - embedding_drift   cosine distance between query and retrieved vectors
  - retrieval_drift   NDCG@k change relative to a baseline score
  - response_drift    lexical similarity (Jaccard) between response batches
  - latency_drift     p95 latency ratio to baseline
  - coverage_drift    unique doc coverage ratio change

In production, swap this module for driftvane or ragdrift-py.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalTrace:
    """Single retrieval event captured from your RAG pipeline."""
    query_embedding: List[float]
    retrieved_embeddings: List[List[float]]
    response_text: str
    latency_ms: float
    retrieved_doc_ids: List[str]
    relevance_scores: List[float] = field(default_factory=list)


@dataclass
class DriftDimensions:
    embedding_drift: float = 0.0
    retrieval_drift: float = 0.0
    response_drift: float = 0.0
    latency_drift: float = 0.0
    coverage_drift: float = 0.0

    def max(self) -> float:
        return max(
            self.embedding_drift,
            self.retrieval_drift,
            self.response_drift,
            self.latency_drift,
            self.coverage_drift,
        )

    def as_dict(self) -> dict:
        return {
            "embedding_drift": round(self.embedding_drift, 4),
            "retrieval_drift": round(self.retrieval_drift, 4),
            "response_drift": round(self.response_drift, 4),
            "latency_drift": round(self.latency_drift, 4),
            "coverage_drift": round(self.coverage_drift, 4),
        }


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


def _jaccard(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return 1.0 - intersection / union


def _ndcg(scores: List[float], k: int = 5) -> float:
    """Normalized DCG@k from a relevance score list (higher = more relevant)."""
    top_k = sorted(scores, reverse=True)[:k]
    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(top_k))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(len(top_k)))
    return dcg / ideal if ideal > 0 else 0.0


def compute_drift(
    current: List[RetrievalTrace],
    baseline: Optional[List[RetrievalTrace]] = None,
) -> DriftDimensions:
    """
    Compute drift dimensions for `current` relative to `baseline`.
    If no baseline, drift is computed as internal variance within current.
    """
    if not current:
        return DriftDimensions()

    d = DriftDimensions()

    # ── embedding drift ──────────────────────────────────────────────────────
    emb_dists: List[float] = []
    for trace in current:
        for ret_emb in trace.retrieved_embeddings:
            if len(ret_emb) == len(trace.query_embedding):
                emb_dists.append(_cosine_distance(trace.query_embedding, ret_emb))
    if emb_dists:
        current_emb = statistics.mean(emb_dists)
        if baseline:
            base_dists = [
                _cosine_distance(t.query_embedding, r)
                for t in baseline
                for r in t.retrieved_embeddings
                if len(r) == len(t.query_embedding)
            ]
            baseline_emb = statistics.mean(base_dists) if base_dists else current_emb
            d.embedding_drift = abs(current_emb - baseline_emb)
        else:
            d.embedding_drift = statistics.stdev(emb_dists) if len(emb_dists) > 1 else 0.0

    # ── retrieval drift (NDCG) ───────────────────────────────────────────────
    ndcg_scores = [_ndcg(t.relevance_scores) for t in current if t.relevance_scores]
    if ndcg_scores:
        current_ndcg = statistics.mean(ndcg_scores)
        if baseline:
            base_ndcg_scores = [_ndcg(t.relevance_scores) for t in baseline if t.relevance_scores]
            base_ndcg = statistics.mean(base_ndcg_scores) if base_ndcg_scores else current_ndcg
            d.retrieval_drift = abs(base_ndcg - current_ndcg)
        else:
            d.retrieval_drift = statistics.stdev(ndcg_scores) if len(ndcg_scores) > 1 else 0.0

    # ── response drift (Jaccard) ─────────────────────────────────────────────
    texts = [t.response_text for t in current]
    if len(texts) > 1 and baseline:
        base_texts = [t.response_text for t in baseline]
        pairs = [_jaccard(a, b) for a, b in zip(texts, base_texts)]
        d.response_drift = statistics.mean(pairs)
    elif len(texts) > 1:
        pairs = []
        for i in range(len(texts) - 1):
            pairs.append(_jaccard(texts[i], texts[i + 1]))
        d.response_drift = statistics.mean(pairs) if pairs else 0.0

    # ── latency drift ────────────────────────────────────────────────────────
    latencies = [t.latency_ms for t in current]
    if latencies:
        p95_current = sorted(latencies)[int(len(latencies) * 0.95)]
        if baseline:
            base_lat = [t.latency_ms for t in baseline]
            p95_base = sorted(base_lat)[int(len(base_lat) * 0.95)] if base_lat else p95_current
            d.latency_drift = abs(p95_current - p95_base) / max(p95_base, 1.0)
        else:
            mean_lat = statistics.mean(latencies)
            d.latency_drift = statistics.stdev(latencies) / max(mean_lat, 1.0) if len(latencies) > 1 else 0.0

    # ── coverage drift ───────────────────────────────────────────────────────
    current_docs = {doc for t in current for doc in t.retrieved_doc_ids}
    if baseline:
        base_docs = {doc for t in baseline for doc in t.retrieved_doc_ids}
        all_docs = current_docs | base_docs
        if all_docs:
            d.coverage_drift = len(base_docs.symmetric_difference(current_docs)) / len(all_docs)
    else:
        # Measure coverage breadth vs. first-trace baseline
        per_trace_docs = [set(t.retrieved_doc_ids) for t in current]
        if len(per_trace_docs) > 1:
            all_seen = set()
            fracs = []
            for docs in per_trace_docs:
                before = len(all_seen)
                all_seen |= docs
                fracs.append((len(all_seen) - before) / max(len(all_seen), 1))
            d.coverage_drift = 1.0 - statistics.mean(fracs)

    return d
