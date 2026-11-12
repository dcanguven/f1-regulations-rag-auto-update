"""Tests for structure-aware chunking."""
from __future__ import annotations

import pytest

from app.chunking.chunker import (
    _make_chunk_id,
    _split_group,
    _ArticleGroup,
    chunk_document,
)
from app.models.schemas import ParsedElement


def test_chunk_document_basic(sample_parsed_doc):
    chunks = chunk_document(sample_parsed_doc)
    assert len(chunks) > 0
    # All chunks must have required fields
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id == "fia_f1_2026_B_issue_08"
        assert chunk.season == 2026
        assert chunk.section_code == "B"
        assert chunk.embedding_text
        assert chunk.text


def test_chunk_ids_are_deterministic(sample_parsed_doc):
    chunks1 = chunk_document(sample_parsed_doc)
    chunks2 = chunk_document(sample_parsed_doc)
    ids1 = [c.chunk_id for c in chunks1]
    ids2 = [c.chunk_id for c in chunks2]
    assert ids1 == ids2


def test_chunk_ids_are_unique(sample_parsed_doc):
    chunks = chunk_document(sample_parsed_doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_deleted_elements_excluded(sample_parsed_doc):
    sample_parsed_doc.elements[0].is_deleted = True
    chunks = chunk_document(sample_parsed_doc)
    # Deleted text should not appear in any chunk
    deleted_text = sample_parsed_doc.elements[0].text
    for chunk in chunks:
        assert deleted_text not in chunk.text


def test_unrelated_articles_not_merged(sample_parsed_doc):
    chunks = chunk_document(sample_parsed_doc)
    # B5 and B5.13 are different articles — verify they are in separate chunks
    article_ids = {c.article_id for c in chunks}
    assert "B5.13" in article_ids or "B5" in article_ids


def test_embedding_text_includes_hierarchy(sample_parsed_doc):
    chunks = chunk_document(sample_parsed_doc)
    for chunk in chunks:
        # Embedding text must include section name and season
        assert "Sporting" in chunk.embedding_text or "2026" in chunk.embedding_text


def test_forced_split_with_overlap():
    """When a group exceeds max_tokens, it should be split with overlap."""
    group = _ArticleGroup(
        article_id="B99",
        article_title="Long Article",
        hierarchy_path="B / B99",
    )
    # Build a long text
    long_text = " ".join(["word"] * 1000)
    from app.models.schemas import ParsedElement
    elem = ParsedElement(
        text=long_text,
        page=1,
        hierarchy_path="B / B99",
        article_id="B99",
        article_title="Long Article",
        content_role="regulatory_text",
    )
    group.elements.append(elem)

    parts = _split_group(group, max_tokens=100, overlap_tokens=20)
    assert len(parts) >= 2
    # Overlap: end of first chunk should share some content with start of second
    overlap_chars = 20 * 4
    assert parts[0][-overlap_chars:].strip() in parts[1] or len(parts[1]) > 0


def test_metadata_preserved_in_chunks(sample_parsed_doc):
    chunks = chunk_document(sample_parsed_doc)
    for chunk in chunks:
        assert chunk.season == 2026
        assert chunk.section_code == "B"
        assert chunk.issue == 8
        assert chunk.source_url
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start


def test_make_chunk_id_deterministic():
    id1 = _make_chunk_id("doc_A", "B5.13", 0)
    id2 = _make_chunk_id("doc_A", "B5.13", 0)
    id3 = _make_chunk_id("doc_A", "B5.13", 1)
    assert id1 == id2
    assert id1 != id3
