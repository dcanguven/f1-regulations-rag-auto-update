"""Tests for the FastAPI endpoints, including isolation from FIA during queries."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import QueryScope

client = TestClient(app)

# routes.py imports `retrieve` directly, so patch it there
_RETRIEVE = "app.api.routes.retrieve_multi"
_GENERATE = "app.generation.answer.generate_structured"


def _ok(text="Safety car text."):
    return {"can_answer": True, "claims": [{"text": text, "sources": [1]}]}



def _fake_result(candidates=None, *, season=2026, section=None, exact_article=None):
    r = MagicMock()
    r.candidates = candidates or []
    r.scope = QueryScope(season=season, section_code=section, exact_article=exact_article)
    r.was_exact_lookup = False
    r.latency_ms = {}
    r.year_resolution = None
    r.exact_total_chunks = 0
    r.exact_truncated = False
    return r


def _fake_candidate(article_id="B5.13", text="Safety car text."):
    return {
        "id": f"chunk_{article_id}",
        "score": 0.9,
        "payload": {
            "season": 2026,
            "section_code": "B",
            "issue": 8,
            "article_id": article_id,
            "article_title": f"Title {article_id}",
            "page_start": 2,
            "page_end": 3,
            "source_url": "https://www.fia.com/test.pdf",
            "text": text,
            "embedding_text": "...",
        },
    }


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /ask never contacts FIA ────────────────────────────────────────────────────

def test_ask_does_not_call_fia():
    """The /ask endpoint must never trigger FIA downloads."""
    import httpx
    original_get = httpx.get
    fia_urls_called = []

    def spy_get(url, **kwargs):
        if "fia.com" in url:
            fia_urls_called.append(url)
            raise AssertionError(f"FIA was called during /ask: {url}")
        return original_get(url, **kwargs)

    fake_result = _fake_result([_fake_candidate()])

    with patch(_RETRIEVE, return_value=fake_result):
        with patch(_GENERATE, return_value=_ok()):
            with patch("httpx.get", side_effect=spy_get):
                resp = client.post("/ask", json={"question": "What is the safety car rule?"})

    assert resp.status_code == 200
    assert fia_urls_called == [], f"FIA was contacted: {fia_urls_called}"


# ── /ask response shape ────────────────────────────────────────────────────────

def test_ask_response_shape():
    fake_result = _fake_result([_fake_candidate()])
    with patch(_RETRIEVE, return_value=fake_result):
        with patch(_GENERATE, return_value=_ok()):
            resp = client.post("/ask", json={"question": "Safety car rules?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "scope_used" in body
    assert "citations" in body
    assert isinstance(body["citations"], list)


def test_ask_debug_mode():
    fake_result = _fake_result()
    with patch(_RETRIEVE, return_value=fake_result):
        resp = client.post("/ask", json={"question": "What is fuel?", "debug": True})
    assert resp.status_code == 200


def test_ask_returns_backend_resolved_citations_and_flags():
    fake_result = _fake_result([_fake_candidate()])
    with patch(_RETRIEVE, return_value=fake_result):
        with patch(_GENERATE, return_value={"can_answer": True, "claims": [{"text": "Safety car text.", "sources": [1, 7]}]}):
            body = client.post("/ask", json={"question": "Safety car rules?"}).json()
    assert body["abstained"] is False
    assert [c["ref"] for c in body["citations"]] == ["C1"]
    assert body["citations"][0]["article_id"] == "B5.13" and body["citations"][0]["page_start"] == 2
    assert body["answer"].endswith("[C1]")


def test_ask_abstains_when_model_declines():
    fake_result = _fake_result([_fake_candidate()])
    with patch(_RETRIEVE, return_value=fake_result):
        with patch(_GENERATE, return_value={"can_answer": False, "claims": []}):
            body = client.post("/ask", json={"question": "Who won in 2023?"}).json()
    assert body["abstained"] is True and body["citations"] == []
    assert body["abstain_reason"] == "model_declined"


def test_ask_short_question_rejected():
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422


# ── Previously indexed data remains queryable ──────────────────────────────────

def test_indexed_data_queryable_when_fia_unavailable():
    """When FIA is down, /ask must still work from Qdrant/registry."""
    fake_result = _fake_result(
        [_fake_candidate("C4.1", "The car must weigh at least 800kg.")],
        section="C",
    )
    with patch(_RETRIEVE, return_value=fake_result):
        with patch(_GENERATE, return_value=_ok("The car must weigh at least 800kg.")):
            with patch("app.ingestion.catalog.fetch_catalog", side_effect=RuntimeError("FIA down")):
                resp = client.post("/ask", json={"question": "What is the minimum weight?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "citations" in body


# ── Scope filtering ────────────────────────────────────────────────────────────

def test_ask_with_section_filter():
    """Section filter must be passed through to retrieve."""
    fake_result = _fake_result(section="C")
    captured = {}

    def capture_retrieve(question, bm25_index, **kwargs):
        captured.update(kwargs)
        return fake_result

    with patch(_RETRIEVE, side_effect=capture_retrieve):
        with patch(_GENERATE, return_value=_ok()):
            client.post("/ask", json={"question": "Minimum weight?", "section": "C"})

    assert captured.get("section_override") == "C"


def test_ask_with_season_filter():
    """Season override must be forwarded correctly."""
    fake_result = _fake_result(season=2026)
    captured = {}

    def capture_retrieve(question, bm25_index, **kwargs):
        captured.update(kwargs)
        return fake_result

    with patch(_RETRIEVE, side_effect=capture_retrieve):
        with patch(_GENERATE, return_value=_ok()):
            client.post("/ask", json={"question": "Rules?", "season": 2026})

    assert captured.get("season_override") == 2026
