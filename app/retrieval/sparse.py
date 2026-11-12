"""
BM25 sparse retrieval.

The BM25 index is built in-memory from documents stored in Qdrant.
It is rebuilt on first use (lazy) and can be explicitly refreshed.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Index:
    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._payloads: list[dict[str, Any]] = []
        self._ids: list[str] = []

    def build(self, payloads: list[dict[str, Any]], ids: list[str]) -> None:
        """Build from payload dicts (must contain 'text' key)."""
        self._payloads = payloads
        self._ids = ids
        corpus = [_tokenize(p.get("text", "")) for p in payloads]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built: %d documents", len(corpus))

    def search(
        self,
        query: str,
        top_k: int,
        season_filter: Optional[int] = None,
        section_filter: Optional[str] = None,
        effective_year_filter: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._payloads:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Gather all with positive score, apply metadata filter
        candidates: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            p = self._payloads[idx]
            if season_filter is not None and p.get("season") != season_filter:
                continue
            if section_filter is not None and p.get("section_code") != section_filter:
                continue
            if effective_year_filter is not None and p.get("effective_year") != effective_year_filter:
                continue
            candidates.append((score, idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in candidates[:top_k]:
            results.append({
                "score": float(score),
                "payload": self._payloads[idx],
                "id": self._ids[idx],
            })
        return results

    def is_built(self) -> bool:
        return self._bm25 is not None


_global_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    global _global_index
    if _global_index is None:
        _global_index = BM25Index()
    return _global_index


def build_bm25_from_qdrant(client, collection: str) -> BM25Index:
    """Scroll all chunks from Qdrant and build BM25 index."""
    from app.storage.qdrant_store import scroll_by_document

    # We need all payloads — use generic scroll
    from qdrant_client.http import models as qmodels

    payloads: list[dict] = []
    ids: list[str] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
        )
        for r in records:
            payloads.append(r.payload)
            ids.append(str(r.id))
        if next_offset is None:
            break
        offset = next_offset

    idx = get_bm25_index()
    idx.build(payloads, ids)
    return idx
