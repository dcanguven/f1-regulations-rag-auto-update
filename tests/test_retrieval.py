"""Tests for query processing, RRF, and citation validation."""
from __future__ import annotations

import pytest

from app.generation.citation import resolve_citations
from app.retrieval.query_processor import extract_scope, normalize_article
from app.retrieval.rrf import rrf_fuse


# ── Article normalization ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("B5.13", "B5.13"),
    ("b 5.13", "B5.13"),
    ("B 5 13", "B5.13"),
    ("article b5.13", "B5.13"),
    ("Article B5.13.4", "B5.13.4"),
    ("Madde B5.13", "B5.13"),
    ("C14.3", "C14.3"),
    ("section C 14", "C14"),
])
def test_normalize_article(raw, expected):
    result = normalize_article(raw)
    assert result == expected, f"normalize_article({raw!r}) = {result!r}, expected {expected!r}"


def test_normalize_article_none():
    assert normalize_article("what is the safety car rule") is None


# ── Scope extraction ───────────────────────────────────────────────────────────

def test_extract_scope_season():
    scope = extract_scope("In the 2026 season, what are the rules for the safety car?")
    assert scope.season == 2026


def test_extract_scope_defaults_to_2026():
    scope = extract_scope("What is the minimum weight?")
    assert scope.season == 2026


def test_extract_scope_section():
    scope = extract_scope("According to Section B, how many cars can a team enter?")
    assert scope.section_code == "B"


def test_extract_scope_exact_article():
    scope = extract_scope("What does article B5.13 say?")
    assert scope.exact_article == "B5.13"
    assert scope.section_code == "B"


def test_extract_scope_turkish():
    scope = extract_scope("B5.13 kuralı nedir?")
    assert scope.exact_article == "B5.13"


def test_extract_scope_no_contamination():
    """Season from one question should not bleed into another."""
    scope1 = extract_scope("What is the 2026 rule for pit stops?")
    scope2 = extract_scope("What is the minimum weight?")
    assert scope1.season == 2026
    assert scope2.season == 2026  # default


# ── RRF ───────────────────────────────────────────────────────────────────────

def _make_result(doc_id: str, score: float) -> dict:
    return {"id": doc_id, "score": score, "payload": {"text": f"text for {doc_id}"}}


def test_rrf_fuse_deduplicates():
    dense = [_make_result("A", 0.9), _make_result("B", 0.8)]
    sparse = [_make_result("A", 10.0), _make_result("C", 8.0)]
    fused = rrf_fuse(dense, sparse, top_k=10, k=60)
    ids = [r["id"] for r in fused]
    assert ids.count("A") == 1  # no duplicates


def test_rrf_fuse_top_k():
    dense = [_make_result(f"D{i}", 1.0 / i) for i in range(1, 21)]
    sparse = [_make_result(f"S{i}", 10.0 / i) for i in range(1, 21)]
    fused = rrf_fuse(dense, sparse, top_k=5, k=60)
    assert len(fused) == 5


def test_rrf_fuse_ranks_common_docs_higher():
    """A doc appearing in both lists should score higher than one in only one."""
    dense = [_make_result("shared", 0.9), _make_result("dense_only", 0.8)]
    sparse = [_make_result("shared", 8.0), _make_result("sparse_only", 7.0)]
    fused = rrf_fuse(dense, sparse, top_k=10, k=60)
    ids = [r["id"] for r in fused]
    assert ids[0] == "shared"


def test_rrf_fuse_empty_lists():
    result = rrf_fuse([], [], top_k=5)
    assert result == []


def test_rrf_fuse_one_empty():
    dense = [_make_result("A", 0.9)]
    result = rrf_fuse(dense, [], top_k=5)
    assert len(result) == 1
    assert result[0]["id"] == "A"


# ── Citation validation ────────────────────────────────────────────────────────

def _make_candidate(article_id: str, page: int = 1) -> dict:
    return {
        "id": f"chunk_{article_id}",
        "payload": {
            "season": 2026,
            "section_code": "B",
            "issue": 8,
            "article_id": article_id,
            "article_title": f"Title for {article_id}",
            "page_start": page,
            "page_end": page,
            "source_url": "https://www.fia.com/test.pdf",
        },
    }


def test_citation_invalid_index_ignored():
    citations = resolve_citations([99], {1: _make_candidate("B5.13")["payload"]})
    assert citations == []


def test_citation_no_invented_data():
    evidence = {1: _make_candidate("B5.13", page=42)["payload"]}
    citations = resolve_citations([1, 1, 99], evidence)
    assert len(citations) == 1
    assert (citations[0].ref, citations[0].article_id, citations[0].page_start) == ("C1", "B5.13", 42)
    assert citations[0].source_url == "https://www.fia.com/test.pdf"
