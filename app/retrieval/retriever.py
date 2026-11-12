"""
Main retrieval pipeline.

Orchestrates: exact lookup OR dense+sparse+RRF → rerank → context expansion.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import settings
from app.models.schemas import QueryScope
from app.retrieval.context_expander import expand_context
from app.retrieval.embedder import Embedder
from app.retrieval.query_processor import extract_scope, is_exact_article_query, split_question
from app.retrieval.reranker import Reranker
from app.retrieval.rrf import rrf_fuse
from app.retrieval.sparse import BM25Index
from app.storage.qdrant_store import (
    get_qdrant_client,
    known_article_ids,
    scroll_by_article,
    search_dense,
)

logger = logging.getLogger(__name__)

YEAR_MENTION_WINDOW = 3


def _approx_tokens(payload: dict[str, Any]) -> int:
    return max(1, len(payload.get("text", "")) // 4)


_LIST_LABEL_RE = re.compile(r"^\(?[a-z]{1,3}(?:\.[a-z]+)?[\.\)]?\s")
SHORT_CHUNK_CHARS = 260
LIST_ARTICLE_SIBLINGS = 4


def complete_articles(
    fused: list[dict[str, Any]],
    qdrant,
    query_vector: list[float],
    effective_year: Optional[int],
    section_code: Optional[str],
    max_articles: int = 2,
    per_article: int = 1,
    list_per_article: int = 3,
) -> list[dict[str, Any]]:
    seen = {str(c["id"]) for c in fused}
    articles: list[str] = []
    for c in fused:
        aid = c["payload"].get("article_id")
        if aid and aid not in articles:
            articles.append(aid)
        if len(articles) >= max_articles:
            break
    for c in fused:
        aid = c["payload"].get("article_id")
        if aid and aid not in articles and len(c["payload"].get("text", "")) < SHORT_CHUNK_CHARS:
            articles.append(aid)
    extra: list[dict[str, Any]] = []
    for rank, aid in enumerate(articles):
        siblings = scroll_by_article(
            qdrant, settings.qdrant_collection, aid, section_code=section_code,
            effective_year=effective_year, descendants=False, limit=50, with_vectors=True,
        )
        take = list_per_article if rank == 0 and len(siblings) > LIST_ARTICLE_SIBLINGS else per_article
        fresh = [x for x in siblings if str(x["id"]) not in seen]
        fresh.sort(key=lambda x: -sum(a * b for a, b in zip(query_vector, x["vector"])))
        chosen = fresh[:take]
        if take > per_article:
            lead = [x for x in fresh if not _LIST_LABEL_RE.match(x["payload"].get("text", ""))]
            if lead and lead[0] not in chosen:
                chosen = [lead[0]] + chosen[: take - 1]
        for item in chosen:
            item.pop("vector", None)
            seen.add(str(item["id"]))
            extra.append(item)
    return fused + extra


def maybe_expand(candidates: list[dict[str, Any]], qdrant) -> list[dict[str, Any]]:
    if not settings.enable_context_expansion:
        return candidates
    return expand_context(candidates, qdrant, settings.qdrant_collection)


def validate_scope(scope: QueryScope, qdrant) -> QueryScope:
    if scope.exact_article and scope.exact_article not in known_article_ids(qdrant, settings.qdrant_collection):
        logger.info("Ignoring unknown article reference %s", scope.exact_article)
        scope.exact_article = None
        if scope.section_from_article:
            scope.section_code = None
            scope.section_from_article = False
    return scope


def select_exact_context(
    hits: list[dict[str, Any]],
    article_id: str,
    query_vector: Optional[list[float]],
    max_chunks: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], bool]:
    total_tokens = sum(_approx_tokens(h["payload"]) for h in hits)
    if len(hits) <= max_chunks and total_tokens <= max_tokens:
        return hits, False

    def similarity(h: dict[str, Any]) -> float:
        vec = h.get("vector")
        if query_vector is None or vec is None:
            return 0.0
        return float(sum(a * b for a, b in zip(query_vector, vec)))

    own = [h for h in hits if h["payload"].get("article_id") == article_id]
    rest = sorted(
        (h for h in hits if h["payload"].get("article_id") != article_id),
        key=lambda h: (-similarity(h), h["payload"].get("page_start", 0)),
    )
    chosen: list[dict[str, Any]] = []
    used = 0
    for h in own + rest:
        tokens = _approx_tokens(h["payload"])
        if chosen and (len(chosen) >= max_chunks or used + tokens > max_tokens):
            continue
        chosen.append(h)
        used += tokens
    order = {id(h): i for i, h in enumerate(hits)}
    chosen.sort(key=lambda h: order[id(h)])
    return chosen, True


@dataclass
class YearResolution:
    mode: str
    year: Optional[int] = None
    reason: str = ""
    shared_future: list[dict[str, Any]] = field(default_factory=list)
    explicit: bool = False


def decide_year_scope(
    year: int,
    current: list[dict[str, Any]],
    future: list[dict[str, Any]],
) -> YearResolution:
    pattern = re.compile(rf"\b{year}\b")
    mentions = [c for c in current[:YEAR_MENTION_WINDOW] if pattern.search(c["payload"].get("text", ""))]
    if mentions:
        current_articles = {c["payload"].get("article_id") for c in mentions if c["payload"].get("article_id")}
        shared = [f for f in future if f["payload"].get("article_id") in current_articles]
        if shared:
            return YearResolution(
                "ambiguous_both", year,
                f"current regulation text mentions {year} and an approved {year} change exists for the same article",
                shared,
            )
        return YearResolution("current", year, f"current regulation text mentions {year}")
    if future:
        return YearResolution("future", year, f"no current regulation text mentions {year}; approved {year} provisions exist")
    return YearResolution("unresolved", year, f"neither current text nor approved provisions mention {year}")


@dataclass
class RetrievalResult:
    candidates: list[dict[str, Any]]
    scope: QueryScope
    was_exact_lookup: bool
    latency_ms: dict[str, float] = field(default_factory=dict)
    exact_total_chunks: int = 0
    exact_truncated: bool = False
    year_resolution: Optional[YearResolution] = None


def retrieve(
    question: str,
    bm25_index: BM25Index,
    season_override: Optional[int] = None,
    section_override: Optional[str] = None,
    issue_override: Optional[int] = None,
    enable_reranker: Optional[bool] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    rrf_top_k: Optional[int] = None,
    rerank_top_k: Optional[int] = None,
    complete_evidence: bool = False,
) -> RetrievalResult:
    t0 = time.perf_counter()

    # ── Scope ──────────────────────────────────────────────────────────────────
    scope = extract_scope(question)
    if season_override is not None:
        scope.season = season_override
        scope.year_mentioned = season_override
        scope.year_mode = (
            "current" if season_override == settings.default_season
            else "future_explicit" if season_override > settings.default_season else "past"
        )
    if section_override is not None:
        scope.section_code = section_override
    if issue_override is not None:
        scope.issue = issue_override

    cfg_dense = dense_top_k or settings.dense_top_k
    cfg_sparse = sparse_top_k or settings.sparse_top_k
    cfg_rrf = rrf_top_k or settings.rrf_top_k
    cfg_rerank = rerank_top_k or settings.rerank_top_k
    cfg_use_reranker = enable_reranker if enable_reranker is not None else settings.enable_reranker

    qdrant = get_qdrant_client()
    latency: dict[str, float] = {}

    validate_scope(scope, qdrant)

    # ── Exact article lookup ───────────────────────────────────────────────────
    if is_exact_article_query(scope) and scope.exact_article:
        t1 = time.perf_counter()
        hits = scroll_by_article(
            qdrant,
            settings.qdrant_collection,
            scope.exact_article,
            section_code=scope.section_code,
            effective_year=settings.default_season if scope.year_mode == "year_mention" else scope.season,
            limit=settings.exact_fetch_limit,
            with_vectors=True,
        )
        total = len(hits)
        candidates, truncated = select_exact_context(
            hits,
            scope.exact_article,
            None,
            settings.exact_context_max_chunks,
            settings.exact_context_max_tokens,
        )
        if truncated:
            query_vector = Embedder().encode_query(question)
            candidates, truncated = select_exact_context(
                hits,
                scope.exact_article,
                query_vector,
                settings.exact_context_max_chunks,
                settings.exact_context_max_tokens,
            )
            logger.warning(
                "Exact lookup %s truncated to %d of %d chunks", scope.exact_article, len(candidates), total
            )
        for c in candidates:
            c.pop("vector", None)
        latency["exact_lookup_ms"] = (time.perf_counter() - t1) * 1000

        if candidates:
            logger.info("Exact lookup for %s: %d chunks", scope.exact_article, len(candidates))
            candidates = maybe_expand(candidates, qdrant)
            return RetrievalResult(
                candidates=candidates,
                scope=scope,
                was_exact_lookup=True,
                latency_ms=latency,
                exact_total_chunks=total,
                exact_truncated=truncated,
            )
        logger.info("Exact article %s not found, falling back to hybrid", scope.exact_article)

    # ── Dense + sparse retrieval ───────────────────────────────────────────────
    embedder = Embedder()
    t1 = time.perf_counter()
    query_vector = embedder.encode_query(question)
    latency["embed_ms"] = (time.perf_counter() - t1) * 1000

    def hybrid(effective_year: Optional[int]):
        dense = search_dense(
            qdrant, settings.qdrant_collection, query_vector, cfg_dense,
            section_filter=scope.section_code, effective_year_filter=effective_year,
        )
        sparse = bm25_index.search(
            question, cfg_sparse, section_filter=scope.section_code, effective_year_filter=effective_year,
        ) if bm25_index.is_built() else []
        return dense, sparse, rrf_fuse(dense, sparse, cfg_rrf)

    year_resolution: Optional[YearResolution] = None
    t1 = time.perf_counter()
    if scope.year_mode == "year_mention" and scope.year_mentioned:
        year = scope.year_mentioned
        _, _, current_fused = hybrid(settings.default_season)
        _, _, future_fused = hybrid(year)
        year_resolution = decide_year_scope(year, current_fused[:cfg_rerank], future_fused[:cfg_rerank])
        if year_resolution.mode == "future":
            scope.season = year
            dense_results, sparse_results, fused = hybrid(year)
        else:
            scope.season = settings.default_season
            dense_results, sparse_results, fused = hybrid(settings.default_season)
    else:
        dense_results, sparse_results, fused = hybrid(scope.season)
        if scope.year_mode == "future_explicit":
            year_resolution = YearResolution("future", scope.year_mentioned, "explicit question about approved future changes", explicit=True)
        elif scope.year_mode == "past":
            year_resolution = YearResolution("past", scope.year_mentioned, "only 2026 regulations are indexed")
    latency["dense_ms"] = (time.perf_counter() - t1) * 1000
    latency["sparse_ms"] = 0.0

    # ── Rerank ─────────────────────────────────────────────────────────────────
    if cfg_use_reranker and fused:
        t1 = time.perf_counter()
        reranker = Reranker()
        fused = reranker.rerank(question, fused, cfg_rerank)
        latency["rerank_ms"] = (time.perf_counter() - t1) * 1000
    else:
        fused = fused[:cfg_rerank]

    if complete_evidence and fused:
        fused = complete_articles(fused, qdrant, query_vector, scope.season, scope.section_code)

    # ── Context expansion ──────────────────────────────────────────────────────
    fused = maybe_expand(fused, qdrant)
    if year_resolution and year_resolution.mode == "ambiguous_both":
        seen = {str(c["id"]) for c in fused}
        fused = fused + [f for f in year_resolution.shared_future if str(f["id"]) not in seen]

    latency["total_ms"] = (time.perf_counter() - t0) * 1000
    logger.info(
        "Retrieved %d candidates in %.0fms (dense=%d sparse=%d)",
        len(fused), latency["total_ms"], len(dense_results), len(sparse_results),
    )

    return RetrievalResult(
        candidates=fused,
        scope=scope,
        was_exact_lookup=False,
        latency_ms=latency,
        year_resolution=year_resolution,
    )


MULTI_PART_TOP_K = 3


def retrieve_multi(question: str, bm25_index: BM25Index, **kwargs) -> RetrievalResult:
    parts = split_question(question)
    if len(parts) < 2:
        return retrieve(question, bm25_index, complete_evidence=True, **kwargs)
    kwargs.pop("rerank_top_k", None)
    results = [
        retrieve(part, bm25_index, rerank_top_k=MULTI_PART_TOP_K, complete_evidence=True, **kwargs) for part in parts
    ]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank in range(max(len(r.candidates) for r in results)):
        for r in results:
            if rank < len(r.candidates):
                cand = r.candidates[rank]
                if str(cand["id"]) not in seen:
                    seen.add(str(cand["id"]))
                    merged.append(cand)
    first = results[0]
    return RetrievalResult(
        candidates=merged,
        scope=first.scope,
        was_exact_lookup=any(r.was_exact_lookup for r in results),
        latency_ms={"total_ms": sum(r.latency_ms.get("total_ms", 0.0) for r in results)},
        exact_total_chunks=max(r.exact_total_chunks for r in results),
        exact_truncated=any(r.exact_truncated for r in results),
        year_resolution=next((r.year_resolution for r in results if r.year_resolution), None),
    )
