from __future__ import annotations

from typing import Any

from app.models.schemas import Citation


def citation_from_payload(ref: str, payload: dict[str, Any]) -> Citation:
    return Citation(
        season=payload.get("season", 0),
        section=payload.get("section_code", ""),
        issue=payload.get("issue", 0),
        article_id=payload.get("article_id"),
        article_title=payload.get("article_title"),
        page_start=payload.get("page_start", 0),
        page_end=payload.get("page_end", 0),
        source_url=payload.get("source_url", ""),
        ref=ref,
        section_name=payload.get("section_name", ""),
        document_id=payload.get("document_id", ""),
        effective_year=payload.get("effective_year"),
        is_current=payload.get("is_current", True),
        is_binding=payload.get("is_binding", True),
    )


def resolve_citations(
    indexes: list[int],
    evidence: dict[int, dict[str, Any]],
) -> list[Citation]:
    return [citation_from_payload(f"C{i}", evidence[i]) for i in sorted(set(indexes)) if i in evidence]
