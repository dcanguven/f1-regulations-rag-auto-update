from __future__ import annotations

import hashlib
import threading
import time
from unittest.mock import patch

import pytest

from app.config import settings
from app.ingestion import updater
from app.ingestion.catalog import CatalogEntry
from app.models.schemas import Chunk, DeletedProvision
from app.storage import qdrant_store as qs
from app.storage.index_state import read_generation
from app.storage.sqlite_store import SQLiteStore

DIM = 8


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [[1.0 if i == j % DIM else 0.0 for i in range(DIM)] for j, _ in enumerate(texts)]


def entry(section="B", issue=9, date="2026-10-01", url=None):
    return CatalogEntry(2026, section, "Sporting", issue, date, url or f"https://fia.test/{section}_{issue}.pdf")


def make_chunks(doc_id, section, issue, n=4, future=True):
    out = []
    for i in range(n):
        is_future = future and i == n - 1
        out.append(Chunk(
            chunk_id=hashlib.md5(f"{doc_id}-{i}".encode()).hexdigest(),
            document_id=doc_id, season=2026, section_code=section, section_name="Sporting", issue=issue,
            article_id=f"{section}{i + 1}.1", article_title="t", hierarchy_path=f"{section}",
            page_start=i + 1, page_end=i + 1, content_type="article", content_role="regulatory_text",
            source_url="u", text=f"text {doc_id} {i}", embedding_text=f"text {doc_id} {i}",
            article_path=[f"{section}{i + 1}"], effective_year=2027 if is_future else 2026,
            is_current=not is_future, pipeline_version=settings.pipeline_version,
        ))
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, value in {
        "qdrant_path": str(tmp_path / "qdrant"), "db_path": str(tmp_path / "reg.db"), "data_dir": str(tmp_path / "data"),
        "qdrant_collection": "t", "embedding_dim": DIM, "log_dir": str(tmp_path / "logs"), "qdrant_lock_wait_seconds": 0.6,
    }.items():
        monkeypatch.setattr(settings, name, value)
    downloads = []

    def fake_download(url, dest, timeout=60.0):
        downloads.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-" + url.encode())
        return hashlib.sha256(b"%PDF-" + url.encode()).hexdigest()

    def fake_parse(pdf_path, document_id, season, section_code, section_name, issue, source_url):
        return (document_id, section_code, issue)

    monkeypatch.setattr(updater, "download_pdf", fake_download)
    monkeypatch.setattr(updater, "parse_pdf", fake_parse)
    monkeypatch.setattr(updater, "chunk_document", lambda p: make_chunks(*p))
    monkeypatch.setattr(
        updater, "extract_deleted",
        lambda p: [DeletedProvision(p[0], 2026, p[1], p[2], 3, f"{p[1]}9.9", None, 2026, f"deleted in issue {p[2]}")],
    )
    return {"downloads": downloads, "embedder": FakeEmbedder()}


def run(env, entries, **kw):
    return updater.run_update(embedder=env["embedder"], fetch=lambda: entries, **kw)


def points(section="B"):
    client = qs.get_qdrant_client()
    try:
        return qs.section_document_ids(client, "t", 2026, section), client.count("t", exact=True).count
    finally:
        client.close()


def test_first_index_then_no_change_touches_nothing(env):
    r = run(env, [entry(issue=8)])
    assert r.status == "ok" and r.new == ["fia_f1_2026_B_issue_08"]
    gen = read_generation()
    downloads, embeds = len(env["downloads"]), env["embedder"].calls
    with patch.object(updater, "get_qdrant_client", side_effect=AssertionError("qdrant must not be opened")):
        r2 = run(env, [entry(issue=8)])
    assert r2.status == "no_changes" and not r2.planned
    assert len(env["downloads"]) == downloads and env["embedder"].calls == embeds
    assert read_generation() == gen
    with SQLiteStore(settings.db_file) as db:
        assert db.get_last_successful_check() is not None


def test_new_issue_replaces_superseded_vectors_and_keeps_history(env):
    run(env, [entry(issue=8)])
    r = run(env, [entry(issue=9, date="2026-11-01")])
    assert r.status == "ok" and r.new == ["fia_f1_2026_B_issue_09"]
    ids, total = points()
    assert ids == {"fia_f1_2026_B_issue_09"} and total == 4
    client = qs.get_qdrant_client()
    try:
        future = qs.scroll_by_article(client, "t", "B4.1", effective_year=2027, descendants=False)
        current = qs.scroll_by_article(client, "t", "B4.1", effective_year=2026, descendants=False)
        assert future and future[0]["payload"]["is_current"] is False and future[0]["payload"]["issue"] == 9
        assert not current
        assert qs.scroll_by_article(client, "t", "B1.1", effective_year=2026, descendants=False)
    finally:
        client.close()
    with SQLiteStore(settings.db_file) as db:
        assert db.get_indexed_document(2026, "B").document_id == "fia_f1_2026_B_issue_09"
        assert not db.get_document("fia_f1_2026_B_issue_08").is_latest_issue
        assert db.get_document("fia_f1_2026_B_issue_08").status == "superseded"
        assert db.list_deleted_provisions("fia_f1_2026_B_issue_08")
        assert db.list_deleted_provisions("fia_f1_2026_B_issue_09")
        assert not db.journal_open()


def test_only_changed_sections_are_downloaded(env):
    run(env, [entry("A", 3), entry("B", 8)])
    before = len(env["downloads"])
    r = run(env, [entry("A", 3), entry("B", 9)])
    assert r.new == ["fia_f1_2026_B_issue_09"] and len(env["downloads"]) == before + 1
    assert points("A")[0] == {"fia_f1_2026_A_issue_03"}


def test_catalog_failure_preserves_index(env):
    run(env, [entry(issue=8)])

    def boom():
        raise RuntimeError("FIA down")

    r = updater.run_update(embedder=env["embedder"], fetch=boom)
    assert r.status == "failed" and "catalog" in r.failures
    assert points()[0] == {"fia_f1_2026_B_issue_08"}


def test_failed_processing_keeps_previous_issue(env, monkeypatch):
    run(env, [entry(issue=8)])
    monkeypatch.setattr(updater, "download_pdf", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("download broke")))
    r = run(env, [entry(issue=9)])
    assert r.status == "failed" and "fia_f1_2026_B_issue_09" in r.failures
    assert points()[0] == {"fia_f1_2026_B_issue_08"}
    with SQLiteStore(settings.db_file) as db:
        assert db.get_indexed_document(2026, "B").issue == 8


def test_commit_failure_rolls_back_partial_points(env, monkeypatch):
    run(env, [entry(issue=8)])
    real = qs.upsert_chunks

    def partial_then_crash(client, collection, chunks, vectors, batch_size=64):
        real(client, collection, chunks[:2], vectors[:2])
        raise OSError("disk full")

    monkeypatch.setattr(updater, "upsert_chunks", partial_then_crash)
    r = run(env, [entry(issue=9)])
    assert r.status == "failed"
    assert points()[0] == {"fia_f1_2026_B_issue_08"}
    with SQLiteStore(settings.db_file) as db:
        assert db.get_indexed_document(2026, "B").issue == 8
        assert not db.journal_open()
    monkeypatch.setattr(updater, "upsert_chunks", real)
    r2 = run(env, [entry(issue=9)])
    assert r2.status == "ok" and points()[0] == {"fia_f1_2026_B_issue_09"}


def test_hard_kill_before_activation_is_recovered(env):
    run(env, [entry(issue=8)])
    new_id = "fia_f1_2026_B_issue_09"
    client = qs.get_qdrant_client()
    chunks = make_chunks(new_id, "B", 9)
    qs.upsert_chunks(client, "t", chunks[:2], FakeEmbedder().encode(["a", "b"]))
    client.close()
    with SQLiteStore(settings.db_file) as db:
        db.journal_start(new_id, 2026, "B", False, 4)
    assert points()[0] == {"fia_f1_2026_B_issue_08", new_id}
    r = run(env, [entry(issue=9)])
    assert r.status == "ok" and r.recovered >= 1 and r.new == [new_id]
    ids, total = points()
    assert ids == {new_id} and total == 4


def test_hard_kill_after_activation_finishes_cleanup(env):
    run(env, [entry(issue=8)])
    new_id, old_id = "fia_f1_2026_B_issue_09", "fia_f1_2026_B_issue_08"
    prepared = updater.prepare_section(
        updater.SectionPlan("B", entry(issue=9), new_id, "new_issue"), SQLiteStore(settings.db_file).connect(), env["embedder"]
    )
    client = qs.get_qdrant_client()
    qs.upsert_chunks(client, "t", prepared.chunks, prepared.vectors)
    client.close()
    with SQLiteStore(settings.db_file) as db:
        jid = db.journal_start(new_id, 2026, "B", False, 4)
        from app.models.schemas import SourceDocument
        doc = SourceDocument(new_id, 2026, "B", "Sporting", 9, "active", "2026-10-01", "u", None, None, True, None, None)
        db.activate_document(doc, "h", "p", prepared.provisions, jid)
    assert points()[0] == {old_id, new_id}
    r = run(env, [entry(issue=9)])
    assert r.recovered >= 1 and points()[0] == {new_id}
    with SQLiteStore(settings.db_file) as db:
        assert not db.journal_open()


def test_same_issue_with_new_publication_date_and_identical_file_is_not_reindexed(env):
    run(env, [entry(issue=8, url="https://fia.test/a.pdf")])
    embeds = env["embedder"].calls
    with SQLiteStore(settings.db_file) as db:
        db.conn.execute("UPDATE documents SET document_hash=?", (hashlib.sha256(b"%PDF-https://fia.test/b.pdf").hexdigest(),))
        db.conn.commit()
    r = run(env, [entry(issue=8, url="https://fia.test/b.pdf")])
    assert r.status == "ok" and not r.new and not r.updated and r.unchanged == ["fia_f1_2026_B_issue_08"]
    assert env["embedder"].calls == embeds


def test_same_issue_with_changed_content_is_updated_in_place(env):
    run(env, [entry(issue=8, url="https://fia.test/a.pdf")])
    r = run(env, [entry(issue=8, url="https://fia.test/changed.pdf")])
    assert r.updated == ["fia_f1_2026_B_issue_08"]
    assert points() == ({"fia_f1_2026_B_issue_08"}, 4)


def test_dry_run_and_due_gating(env):
    r = run(env, [entry(issue=8)], dry_run=True)
    assert r.status == "dry_run" and r.planned and not env["downloads"]
    run(env, [entry(issue=8)])
    r = run(env, [entry(issue=9)], if_due=True)
    assert r.status == "not_due" and points()[0] == {"fia_f1_2026_B_issue_08"}
    with SQLiteStore(settings.db_file) as db:
        db.conn.execute("UPDATE ingestion_log SET run_at='2020-01-01T00:00:00+00:00'")
        db.conn.commit()
    assert run(env, [entry(issue=9)], if_due=True).status == "ok"


def test_concurrent_update_is_rejected_by_lock(env):
    with updater.update_lock() as first:
        assert first
        r = run(env, [entry(issue=8)])
        assert r.status == "busy"
    assert run(env, [entry(issue=8)]).status == "ok"


def test_commit_waits_for_api_to_release_embedded_qdrant(env, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_lock_wait_seconds", 5.0)
    run(env, [entry(issue=8)])
    holder = qs.get_qdrant_client()
    threading.Timer(1.0, holder.close).start()
    start = time.monotonic()
    r = run(env, [entry(issue=9)])
    assert r.status == "ok" and time.monotonic() - start >= 0.9
    assert points()[0] == {"fia_f1_2026_B_issue_09"}


def test_api_request_waits_for_commit_instead_of_failing(env, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_lock_wait_seconds", 5.0)
    run(env, [entry(issue=8)])
    writer = qs.get_qdrant_client()
    threading.Timer(0.8, writer.close).start()
    start = time.monotonic()
    reader = qs.get_qdrant_client()
    assert time.monotonic() - start >= 0.7
    assert reader.count("t", exact=True).count == 4
    reader.close()


def test_lock_wait_times_out_with_clear_error(env):
    holder = qs.get_qdrant_client()
    try:
        with pytest.raises(RuntimeError, match="already accessed"):
            qs.get_qdrant_client()
    finally:
        holder.close()


def test_api_rebuilds_bm25_after_generation_change(env, monkeypatch):
    from app.api import routes
    from app.retrieval.sparse import get_bm25_index
    run(env, [entry(issue=8)])
    monkeypatch.setattr(routes, "_bm25_generation", None)
    routes._ensure_bm25()
    idx = get_bm25_index()
    assert idx.is_built() and {p["document_id"] for p in idx._payloads} == {"fia_f1_2026_B_issue_08"}
    run(env, [entry(issue=9)])
    routes._ensure_bm25()
    assert {p["document_id"] for p in idx._payloads} == {"fia_f1_2026_B_issue_09"}


def test_empty_catalog_is_a_failed_check_and_keeps_index(env):
    run(env, [entry(issue=8)])
    r = run(env, [])
    assert r.status == "failed" and "catalog" in r.failures
    assert points() == ({"fia_f1_2026_B_issue_08"}, 4)
    with SQLiteStore(settings.db_file) as db:
        assert db.conn.execute("SELECT error FROM ingestion_log ORDER BY id DESC LIMIT 1").fetchone()[0]


def test_older_catalog_issue_never_replaces_newer_index(env):
    run(env, [entry(issue=9)])
    r = run(env, [entry(issue=8)])
    assert r.status == "no_changes" and points()[0] == {"fia_f1_2026_B_issue_09"}


def test_non_pdf_download_is_rejected_and_index_kept(env, monkeypatch):
    run(env, [entry(issue=8)])

    def html_download(url, dest, timeout=60.0):
        dest.write_bytes(b"<html>Service unavailable</html>")
        return "x"

    monkeypatch.setattr(updater, "download_pdf", html_download)
    r = run(env, [entry(issue=9)])
    assert r.status == "failed" and "not a PDF" in r.failures["fia_f1_2026_B_issue_09"]
    assert points()[0] == {"fia_f1_2026_B_issue_08"}


def test_suspiciously_small_new_issue_is_refused(env, monkeypatch):
    run(env, [entry(issue=8)])
    monkeypatch.setattr(updater, "chunk_document", lambda p: make_chunks(*p, n=1))
    r = run(env, [entry(issue=9)])
    assert r.status == "failed" and "far below" in r.failures["fia_f1_2026_B_issue_09"]
    assert points() == ({"fia_f1_2026_B_issue_08"}, 4)
    with SQLiteStore(settings.db_file) as db:
        assert not db.journal_open()


def test_interrupted_in_place_update_is_retried_even_without_metadata_change(env, monkeypatch):
    run(env, [entry(issue=8)])
    real = qs.upsert_chunks

    def partial_then_crash(client, collection, chunks, vectors, batch_size=64):
        real(client, collection, chunks[:2], vectors[:2])
        raise OSError("killed")

    monkeypatch.setattr(updater, "upsert_chunks", partial_then_crash)
    assert run(env, [entry(issue=8)], force=True).status == "failed"
    monkeypatch.setattr(updater, "upsert_chunks", real)
    before = env["embedder"].calls
    r = run(env, [entry(issue=8)])
    assert r.updated == ["fia_f1_2026_B_issue_08"] and env["embedder"].calls == before + 1
    assert points() == ({"fia_f1_2026_B_issue_08"}, 4)
    assert run(env, [entry(issue=8)]).status == "no_changes"


def test_ask_returns_503_while_index_is_locked_by_an_update(env, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api import routes
    run(env, [entry(issue=8)])
    monkeypatch.setattr(routes, "_ensure_bm25", lambda: None)
    holder = qs.get_qdrant_client()
    try:
        response = TestClient(app).post("/ask", json={"question": "What is the pit lane speed limit?"})
    finally:
        holder.close()
    assert response.status_code == 503 and "updated" in response.json()["detail"]


def test_frontend_does_not_override_year_detection_with_a_season():
    from unittest.mock import MagicMock
    from streamlit.testing.v1 import AppTest
    captured = {}

    def post(url, json=None, timeout=None):
        captured.update(json)
        m = MagicMock()
        m.ok = True
        m.json.return_value = {"answer": "x [C1]", "scope_used": {}, "citations": [], "claims": [], "abstained": False,
                               "abstain_reason": None, "warnings": [], "year_interpretation": None, "context_truncated": False}
        return m

    with patch("requests.post", post), patch("requests.get", return_value=MagicMock(ok=True)):
        at = AppTest.from_file("frontend/app.py", default_timeout=30).run()
        at.text_input[0].set_value("What will the race distance rule say from 2027?")
        at.button[0].click().run()
    assert captured == {"question": "What will the race distance rule say from 2027?"}
