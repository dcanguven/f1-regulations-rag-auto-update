"""
Context expansion: include parent article header when a sub-clause is retrieved.

We do NOT blindly add neighbors. We only:
  1. Ensure the article heading line is included if it is missing.
  2. For very short chunks (<50 tokens) that are continuations, try to add
     their direct parent text from within the same article.
"""
from __future__ import annotations

from typing import Any

from app.storage.qdrant_store import scroll_by_article


def expand_context(
    candidates: list[dict[str, Any]],
    qdrant_client,
    collection: str,
) -> list[dict[str, Any]]:
    """
    For each candidate, fetch the article heading chunk if not already present.
    Returns the original list (no duplicates added based on chunk_id).
    """
    seen_ids = {str(c["id"]) for c in candidates}
    extra: list[dict[str, Any]] = []

    for cand in candidates:
        payload = cand["payload"]
        article_id = payload.get("article_id")
        if not article_id:
            continue

        # Fetch all chunks for this article to check if heading present
        article_chunks = scroll_by_article(
            qdrant_client,
            collection,
            article_id,
            season=payload.get("season"),
            section_code=payload.get("section_code"),
            effective_year=payload.get("effective_year"),
            descendants=False,
        )

        for ac in article_chunks:
            aid = str(ac["id"])
            if aid in seen_ids:
                continue
            ac_text = ac["payload"].get("text", "")
            # Include heading chunk (contains the article ID + title as first line)
            if article_id in ac_text[:100]:
                extra.append({**ac, "score": cand["score"] * 0.9, "is_context": True})
                seen_ids.add(aid)
                break  # Only add heading once

    return candidates + extra
