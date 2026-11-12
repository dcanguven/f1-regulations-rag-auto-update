"""FastAPI routes: POST /ask, GET /health, GET /status."""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.generation.answer import generate
from app.retrieval.retriever import retrieve_multi
from app.retrieval.sparse import build_bm25_from_qdrant, get_bm25_index
from app.storage.index_state import read_generation
from app.storage.qdrant_store import clear_article_cache, collection_info, get_qdrant_client
from app.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Request / response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    season: Optional[int] = None
    section: Optional[str] = None
    issue: Optional[int] = None
    debug: bool = False


class CitationOut(BaseModel):
    ref: str = ""
    season: int
    section: str
    section_name: str = ""
    issue: int
    article_id: Optional[str] = None
    article_title: Optional[str] = None
    page_start: int
    page_end: int
    source_url: str
    document_id: str = ""
    effective_year: Optional[int] = None
    is_current: bool = True
    is_binding: bool = True


class ClaimOut(BaseModel):
    text: str
    sources: list[str]
    verified: bool = True
    problems: list[str] = []


class AskResponseOut(BaseModel):
    answer: str
    scope_used: dict
    citations: list[CitationOut]
    debug: Optional[dict] = None
    claims: list[ClaimOut] = []
    abstained: bool = False
    abstain_reason: Optional[str] = None
    warnings: list[str] = []
    year_interpretation: Optional[dict] = None
    context_truncated: bool = False


# ── Lazy BM25 boot ─────────────────────────────────────────────────────────────

_bm25_generation: Optional[str] = None


def _ensure_bm25() -> None:
    global _bm25_generation
    generation = read_generation()
    idx = get_bm25_index()
    if idx.is_built() and generation == _bm25_generation:
        return
    try:
        client = get_qdrant_client()
        build_bm25_from_qdrant(client, settings.qdrant_collection)
        clear_article_cache()
        _bm25_generation = generation
    except Exception as e:
        logger.warning("BM25 build failed (will use dense-only): %s", e)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponseOut)
def ask(req: AskRequest) -> AskResponseOut:
    _ensure_bm25()
    t0 = time.perf_counter()

    try:
        result = retrieve_multi(
            question=req.question,
            bm25_index=get_bm25_index(),
            season_override=req.season,
            section_override=req.section.upper() if req.section else None,
            issue_override=req.issue,
        )
    except RuntimeError as e:
        if "already accessed" in str(e):
            raise HTTPException(status_code=503, detail="The regulation index is being updated. Try again in a minute.")
        raise

    retrieval_ms = (time.perf_counter() - t0) * 1000

    response = generate(
        question=req.question,
        candidates=result.candidates,
        scope=result.scope,
        debug=req.debug,
        year_resolution=getattr(result, "year_resolution", None),
        exact_total_chunks=getattr(result, "exact_total_chunks", 0) or 0,
        exact_truncated=bool(getattr(result, "exact_truncated", False)),
    )

    if req.debug and response.debug is not None:
        response.debug["retrieval_ms"] = round(retrieval_ms, 1)
        response.debug["was_exact_lookup"] = result.was_exact_lookup
        response.debug["exact_total_chunks"] = result.exact_total_chunks
        response.debug["exact_truncated"] = result.exact_truncated
        response.debug.update({
            f"retrieval_{k}": round(v, 1)
            for k, v in result.latency_ms.items()
        })

    return AskResponseOut(
        answer=response.answer,
        scope_used=response.scope_used,
        citations=[CitationOut(**vars(c)) for c in response.citations],
        debug=response.debug,
        claims=[ClaimOut(**vars(c)) for c in response.claims],
        abstained=response.abstained,
        abstain_reason=response.abstain_reason,
        warnings=response.warnings,
        year_interpretation=response.year_interpretation,
        context_truncated=response.context_truncated,
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/status")
def status() -> dict:
    # Qdrant status
    try:
        client = get_qdrant_client()
        info = collection_info(client, settings.qdrant_collection)
    except Exception as e:
        info = {"error": str(e)}

    # Registry status
    last_sync = None
    try:
        with SQLiteStore(settings.db_file) as db:
            last_sync = db.get_last_ingestion()
    except Exception:
        pass

    return {
        "qdrant": info,
        "last_sync": last_sync,
        "default_season": settings.default_season,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": settings.enable_reranker,
    }
