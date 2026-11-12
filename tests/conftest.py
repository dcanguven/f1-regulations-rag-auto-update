"""Shared test fixtures."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.models.schemas import ParsedDocument, ParsedElement, SourceDocument


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def sample_source_doc():
    return SourceDocument(
        document_id="fia_f1_2026_B_issue_08",
        season=2026,
        section_code="B",
        section_name="Sporting",
        issue=8,
        status="active",
        publication_date="2026-08-05",
        source_url="https://www.fia.com/test.pdf",
        document_hash=None,
        local_path=None,
        is_latest_issue=True,
        ingested_at=None,
        last_seen_at=None,
    )


@pytest.fixture
def sample_parsed_doc():
    return ParsedDocument(
        document_id="fia_f1_2026_B_issue_08",
        season=2026,
        section_code="B",
        section_name="Sporting",
        issue=8,
        source_url="https://www.fia.com/test.pdf",
        elements=[
            ParsedElement(
                text="B5 RACE PROCEDURE",
                page=1,
                hierarchy_path="B / B5",
                article_id="B5",
                article_title="RACE PROCEDURE",
                content_role="regulatory_text",
            ),
            ParsedElement(
                text="B5.13 Safety Car Procedure",
                page=2,
                hierarchy_path="B / B5 / B5.13",
                article_id="B5.13",
                article_title="Safety Car Procedure",
                content_role="regulatory_text",
            ),
            ParsedElement(
                text="The safety car will be deployed when the clerk of the course considers it necessary.",
                page=2,
                hierarchy_path="B / B5 / B5.13",
                article_id="B5.13",
                article_title="Safety Car Procedure",
                content_role="regulatory_text",
            ),
            ParsedElement(
                text="B5.13.4 All competing cars must form a queue behind the safety car.",
                page=3,
                hierarchy_path="B / B5 / B5.13 / B5.13.4",
                article_id="B5.13.4",
                article_title=None,
                content_role="regulatory_text",
            ),
        ],
    )
