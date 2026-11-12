"""Qdrant vector store — manages the chunk collection."""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.models.schemas import Chunk

logger = logging.getLogger(__name__)

PAYLOAD_FIELDS = [
    "document_id", "season", "section_code", "section_name", "issue",
    "article_id", "article_title", "hierarchy_path", "page_start", "page_end",
    "content_type", "content_role", "source_url", "text", "embedding_text",
    "article_path", "article_heading", "item_path", "appendix_id", "appendix_title",
    "effective_year", "effective_from", "is_current", "is_binding", "contains_changes", "changed_in_issue",
    "contains_table", "governance", "referenced_documents", "pipeline_version",
]

_ARTICLE_CACHE: dict[str, tuple[int, set[str]]] = {}


def clear_article_cache() -> None:
    _ARTICLE_CACHE.clear()


def get_qdrant_client() -> QdrantClient:
    if not settings.qdrant_path:
        return QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    from pathlib import Path
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + settings.qdrant_lock_wait_seconds
    while True:
        try:
            return QdrantClient(path=settings.qdrant_path)
        except RuntimeError as e:
            if "already accessed" not in str(e) or time.monotonic() >= deadline:
                raise
            time.sleep(0.2 + random.random() * 0.2)


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Create collection if it doesn't exist; no-op if it does."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.debug("Collection %s already exists", collection)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    # Payload indices for fast metadata filtering
    keyword_fields = ("section_code", "article_id", "document_id", "article_path", "appendix_id")
    for field in (
        "season", "section_code", "issue", "article_id", "document_id",
        "article_path", "appendix_id", "effective_year",
    ):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD
            if field in keyword_fields
            else qmodels.PayloadSchemaType.INTEGER,
        )
    logger.info("Created collection %s dim=%d", collection, dim)


def chunk_to_point(chunk: Chunk, vector: list[float]) -> qmodels.PointStruct:
    payload = {f: getattr(chunk, f) for f in PAYLOAD_FIELDS}
    return qmodels.PointStruct(
        id=_chunk_id_to_uint(chunk.chunk_id),
        vector=vector,
        payload=payload,
    )


def _chunk_id_to_uint(chunk_id: str) -> str:
    # Qdrant accepts string UUIDs or uint64; we use the string form directly
    # Qdrant v1.x supports string point ids
    return chunk_id


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    batch_size: int = 64,
) -> None:
    points = [chunk_to_point(c, v) for c, v in zip(chunks, vectors)]
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
    logger.info("Upserted %d chunks into %s", len(points), collection)


def document_is_indexed(client: QdrantClient, collection: str, document_id: str) -> bool:
    result = client.count(
        collection_name=collection,
        count_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=document_id)
                )
            ]
        ),
        exact=True,
    )
    return result.count > 0


def _document_filter(document_id: str, *, stale_version: Optional[str] = None) -> qmodels.Filter:
    must = [qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
    must_not = []
    if stale_version is not None:
        must_not.append(
            qmodels.FieldCondition(key="pipeline_version", match=qmodels.MatchValue(value=stale_version))
        )
    return qmodels.Filter(must=must, must_not=must_not or None)


def document_has_points(client: QdrantClient, collection: str, document_id: str) -> bool:
    return client.count(collection_name=collection, count_filter=_document_filter(document_id), exact=True).count > 0


def document_index_current(client: QdrantClient, collection: str, document_id: str, version: str) -> bool:
    total = client.count(collection_name=collection, count_filter=_document_filter(document_id), exact=True).count
    if total == 0:
        return False
    stale = client.count(
        collection_name=collection,
        count_filter=_document_filter(document_id, stale_version=version),
        exact=True,
    ).count
    return stale == 0


def prune_obsolete_chunks(client: QdrantClient, collection: str, document_id: str, keep_ids: set[str]) -> int:
    obsolete: list[Any] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=_document_filter(document_id),
            with_payload=False,
            limit=500,
            offset=offset,
        )
        obsolete.extend(r.id for r in records if str(r.id).replace("-", "") not in keep_ids)
        if offset is None:
            break
    if obsolete:
        client.delete(collection_name=collection, points_selector=qmodels.PointIdsList(points=obsolete))
    return len(obsolete)


def count_document_points(client: QdrantClient, collection: str, document_id: str) -> int:
    return client.count(collection_name=collection, count_filter=_document_filter(document_id), exact=True).count


def _section_filter(season: int, section_code: str, keep_document_id: Optional[str] = None) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(key="season", match=qmodels.MatchValue(value=season)),
        qmodels.FieldCondition(key="section_code", match=qmodels.MatchValue(value=section_code)),
    ]
    must_not = (
        [qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=keep_document_id))]
        if keep_document_id
        else None
    )
    return qmodels.Filter(must=must, must_not=must_not)


def section_document_ids(client: QdrantClient, collection: str, season: int, section_code: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=_section_filter(season, section_code),
            with_payload=["document_id"],
            limit=1000,
            offset=offset,
        )
        ids.update(r.payload.get("document_id") for r in records if r.payload)
        if offset is None:
            return ids


def delete_superseded_documents(
    client: QdrantClient, collection: str, season: int, section_code: str, keep_document_id: str
) -> int:
    selector = _section_filter(season, section_code, keep_document_id)
    stale = client.count(collection_name=collection, count_filter=selector, exact=True).count
    if stale:
        client.delete(collection_name=collection, points_selector=qmodels.FilterSelector(filter=selector))
    return stale


def known_article_ids(client: QdrantClient, collection: str) -> set[str]:
    size = client.count(collection_name=collection, exact=True).count
    cached = _ARTICLE_CACHE.get(collection)
    if cached and cached[0] == size:
        return cached[1]
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection, with_payload=["article_path"], limit=1000, offset=offset
        )
        for r in records:
            ids.update(r.payload.get("article_path") or [])
        if offset is None:
            break
    _ARTICLE_CACHE[collection] = (size, ids)
    return ids


def delete_document_chunks(client: QdrantClient, collection: str, document_id: str) -> None:
    client.delete(
        collection_name=collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
    logger.info("Deleted chunks for document %s", document_id)


def search_dense(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int,
    season_filter: Optional[int] = None,
    section_filter: Optional[str] = None,
    effective_year_filter: Optional[int] = None,
) -> list[dict[str, Any]]:
    must = _build_filter_conditions(season_filter, section_filter, effective_year_filter)
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        query_filter=qmodels.Filter(must=must) if must else None,
        with_payload=True,
    )
    return [{"score": r.score, "payload": r.payload, "id": r.id} for r in response.points]


def scroll_by_article(
    client: QdrantClient,
    collection: str,
    article_id: str,
    season: Optional[int] = None,
    section_code: Optional[str] = None,
    effective_year: Optional[int] = None,
    descendants: bool = True,
    limit: int = 200,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    key = "article_path" if descendants else "article_id"
    must = [qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=article_id))]
    must.extend(_build_filter_conditions(season, section_code, effective_year))
    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qmodels.Filter(must=must),
        with_payload=True,
        with_vectors=with_vectors,
        limit=limit,
    )
    ordered = sorted(
        records,
        key=lambda r: (r.payload.get("page_start", 0), r.payload.get("hierarchy_path", ""), str(r.id)),
    )
    out = []
    for r in ordered:
        item = {"score": 1.0, "payload": r.payload, "id": r.id}
        if with_vectors:
            item["vector"] = r.vector
        out.append(item)
    return out


def scroll_by_document(
    client: QdrantClient,
    collection: str,
    document_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=document_id)
                )
            ]
        ),
        with_payload=True,
        limit=limit,
    )
    return [{"payload": r.payload, "id": r.id} for r in records]


def _build_filter_conditions(
    season: Optional[int],
    section_code: Optional[str],
    effective_year: Optional[int] = None,
) -> list:
    must = []
    if effective_year is not None:
        must.append(
            qmodels.FieldCondition(key="effective_year", match=qmodels.MatchValue(value=effective_year))
        )
    if season is not None:
        must.append(qmodels.FieldCondition(key="season", match=qmodels.MatchValue(value=season)))
    if section_code is not None:
        must.append(qmodels.FieldCondition(key="section_code", match=qmodels.MatchValue(value=section_code)))
    return must


def collection_info(client: QdrantClient, collection: str) -> dict[str, Any]:
    try:
        info = client.get_collection(collection)
        return {
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"error": str(e)}
