"""Reciprocal Rank Fusion for combining dense + sparse results."""
from __future__ import annotations

from typing import Any

from app.config import settings


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    top_k: int,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    Each result dict must contain 'id' and 'payload'.
    Returns deduplicated top_k results sorted by RRF score descending.
    """
    if k is None:
        k = settings.rrf_k

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        doc_id = str(result["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        payloads[doc_id] = result["payload"]

    for rank, result in enumerate(sparse_results, start=1):
        doc_id = str(result["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        payloads[doc_id] = result["payload"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"id": doc_id, "score": score, "payload": payloads[doc_id]}
        for doc_id, score in ranked[:top_k]
    ]
