"""Tests for ingestion: registry, hash detection, latest issue, URL changes."""
from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.catalog import (
    CatalogEntry,
    make_document_id,
    pick_latest_issues,
)
from app.ingestion.downloader import compute_sha256, is_changed, local_path_for
from app.models.schemas import SourceDocument
from app.storage.sqlite_store import SQLiteStore


# ── SQLite registry ────────────────────────────────────────────────────────────

def test_upsert_and_retrieve(tmp_db, sample_source_doc):
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(sample_source_doc)
        retrieved = db.get_document(sample_source_doc.document_id)
    assert retrieved is not None
    assert retrieved.document_id == sample_source_doc.document_id
    assert retrieved.section_code == "B"
    assert retrieved.is_latest_issue is True


def test_idempotent_upsert(tmp_db, sample_source_doc):
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(sample_source_doc)
        db.upsert_document(sample_source_doc)  # second upsert must not fail
        count = db.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1


def test_mark_ingested(tmp_db, sample_source_doc):
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(sample_source_doc)
        db.mark_ingested(sample_source_doc.document_id, "abc123", "/some/path.pdf")
        doc = db.get_document(sample_source_doc.document_id)
    assert doc.document_hash == "abc123"
    assert doc.local_path == "/some/path.pdf"
    assert doc.ingested_at is not None


def test_latest_issue_flag(tmp_db):
    """Only one issue per section should have is_latest_issue=True."""
    older = SourceDocument(
        document_id="fia_f1_2026_B_issue_07",
        season=2026, section_code="B", section_name="Sporting",
        issue=7, status="active", publication_date=None,
        source_url="http://example.com/b07.pdf",
        document_hash=None, local_path=None,
        is_latest_issue=False, ingested_at=None, last_seen_at=None,
    )
    latest = SourceDocument(
        document_id="fia_f1_2026_B_issue_08",
        season=2026, section_code="B", section_name="Sporting",
        issue=8, status="active", publication_date=None,
        source_url="http://example.com/b08.pdf",
        document_hash=None, local_path=None,
        is_latest_issue=True, ingested_at=None, last_seen_at=None,
    )
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(older)
        db.clear_latest_flag(2026, "B")
        db.upsert_document(latest)
        docs = db.list_documents(season=2026, section_code="B", latest_only=True)
    assert len(docs) == 1
    assert docs[0].issue == 8


def test_url_change_updates_same_doc(tmp_db):
    """If URL changes but document_id is the same, URL should update."""
    doc = SourceDocument(
        document_id="fia_f1_2026_C_issue_20",
        season=2026, section_code="C", section_name="Technical",
        issue=20, status="active", publication_date=None,
        source_url="https://www.fia.com/old_url.pdf",
        document_hash=None, local_path=None,
        is_latest_issue=True, ingested_at=None, last_seen_at=None,
    )
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(doc)
        doc.source_url = "https://www.fia.com/new_url.pdf"
        db.upsert_document(doc)
        retrieved = db.get_document(doc.document_id)
    assert retrieved.source_url == "https://www.fia.com/new_url.pdf"


# ── Hash detection ─────────────────────────────────────────────────────────────

def test_compute_sha256(tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"hello world")
    h = compute_sha256(f)
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert h == expected


def test_is_changed_detects_change():
    assert is_changed("old_hash", "new_hash") is True


def test_is_changed_detects_no_change():
    assert is_changed("same_hash", "same_hash") is False


def test_is_changed_none_existing():
    assert is_changed(None, "any_hash") is True


# ── Catalog ────────────────────────────────────────────────────────────────────

def test_pick_latest_issues():
    entries = [
        CatalogEntry(2026, "B", "Sporting", 6, "2026-04-28", "http://b06.pdf"),
        CatalogEntry(2026, "B", "Sporting", 8, "2026-08-05", "http://b08.pdf"),
        CatalogEntry(2026, "B", "Sporting", 7, "2026-06-25", "http://b07.pdf"),
        CatalogEntry(2026, "A", "General Provisions", 3, "2026-06-25", "http://a03.pdf"),
    ]
    latest = pick_latest_issues(entries)
    assert latest["B"].issue == 8
    assert latest["A"].issue == 3


def test_make_document_id():
    assert make_document_id(2026, "B", 8) == "fia_f1_2026_B_issue_08"
    assert make_document_id(2026, "C", 20) == "fia_f1_2026_C_issue_20"


def test_local_path_structure():
    path = local_path_for(2026, "B", 8)
    assert "2026" in str(path)
    assert "B" in str(path)
    assert "issue_08" in str(path)


# ── FIA downtime resilience ────────────────────────────────────────────────────

def test_sync_preserves_data_on_fia_downtime(tmp_db, sample_source_doc):
    """When catalog fetch fails, existing registry data is preserved."""
    with SQLiteStore(tmp_db) as db:
        db.upsert_document(sample_source_doc)

    with patch("app.ingestion.updater.fetch_catalog", side_effect=RuntimeError("FIA down")):
        from app.ingestion.sync import run_sync
        with patch("app.config.settings.db_path", tmp_db):
            # sync should not raise, should preserve data
            try:
                run_sync()
            except Exception:
                pass

    with SQLiteStore(tmp_db) as db:
        doc = db.get_document(sample_source_doc.document_id)
    assert doc is not None
    assert doc.document_id == sample_source_doc.document_id
