"""SQLite source registry — stores document metadata and ingestion state."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.models.schemas import SourceDocument


_CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    season           INTEGER NOT NULL,
    section_code     TEXT NOT NULL,
    section_name     TEXT NOT NULL,
    issue            INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active',
    publication_date TEXT,
    source_url       TEXT NOT NULL,
    document_hash    TEXT,
    local_path       TEXT,
    is_latest_issue  INTEGER NOT NULL DEFAULT 0,
    ingested_at      TEXT,
    last_seen_at     TEXT
);
"""

_CREATE_INGESTION_LOG_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT NOT NULL,
    documents_new INTEGER DEFAULT 0,
    documents_updated INTEGER DEFAULT 0,
    documents_skipped INTEGER DEFAULT 0,
    error         TEXT
);
"""

_CREATE_DELETED_SQL = """
CREATE TABLE IF NOT EXISTS deleted_provisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL,
    season        INTEGER NOT NULL,
    section_code  TEXT NOT NULL,
    issue         INTEGER NOT NULL,
    page          INTEGER NOT NULL,
    article_id    TEXT,
    appendix_id   TEXT,
    effective_year INTEGER,
    text          TEXT NOT NULL
);
"""

_CREATE_JOURNAL_SQL = """
CREATE TABLE IF NOT EXISTS update_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL,
    season          INTEGER NOT NULL,
    section_code    TEXT NOT NULL,
    status          TEXT NOT NULL,
    same_id         INTEGER NOT NULL DEFAULT 0,
    expected_chunks INTEGER,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    detail          TEXT
);
"""


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> "SQLiteStore":
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_DOCUMENTS_SQL)
        self._conn.execute(_CREATE_INGESTION_LOG_SQL)
        self._conn.execute(_CREATE_DELETED_SQL)
        self._conn.execute(_CREATE_JOURNAL_SQL)
        self._conn.commit()
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteStore":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Call connect() first"
        return self._conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> SourceDocument:
        d = dict(row)
        return SourceDocument(
            document_id=d["document_id"],
            season=d["season"],
            section_code=d["section_code"],
            section_name=d["section_name"],
            issue=d["issue"],
            status=d["status"],
            publication_date=d["publication_date"],
            source_url=d["source_url"],
            document_hash=d["document_hash"],
            local_path=d["local_path"],
            is_latest_issue=bool(d["is_latest_issue"]),
            ingested_at=datetime.fromisoformat(d["ingested_at"]) if d["ingested_at"] else None,
            last_seen_at=datetime.fromisoformat(d["last_seen_at"]) if d["last_seen_at"] else None,
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_document(self, document_id: str) -> Optional[SourceDocument]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_latest_document(self, season: int, section_code: str) -> Optional[SourceDocument]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE season=? AND section_code=? AND is_latest_issue=1",
            (season, section_code),
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def list_documents(
        self,
        season: Optional[int] = None,
        section_code: Optional[str] = None,
        latest_only: bool = False,
    ) -> list[SourceDocument]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if season is not None:
            query += " AND season=?"
            params.append(season)
        if section_code is not None:
            query += " AND section_code=?"
            params.append(section_code)
        if latest_only:
            query += " AND is_latest_issue=1"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def get_last_ingestion(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT run_at FROM ingestion_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["run_at"] if row else None

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert_document(self, doc: SourceDocument) -> None:
        now = self._now()
        self.conn.execute(
            """
            INSERT INTO documents
              (document_id, season, section_code, section_name, issue, status,
               publication_date, source_url, document_hash, local_path,
               is_latest_issue, ingested_at, last_seen_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
              season=excluded.season,
              section_code=excluded.section_code,
              section_name=excluded.section_name,
              issue=excluded.issue,
              status=excluded.status,
              publication_date=excluded.publication_date,
              source_url=excluded.source_url,
              document_hash=COALESCE(excluded.document_hash, document_hash),
              local_path=COALESCE(excluded.local_path, local_path),
              is_latest_issue=excluded.is_latest_issue,
              ingested_at=COALESCE(excluded.ingested_at, ingested_at),
              last_seen_at=?
            """,
            (
                doc.document_id, doc.season, doc.section_code, doc.section_name,
                doc.issue, doc.status, doc.publication_date, doc.source_url,
                doc.document_hash, doc.local_path,
                int(doc.is_latest_issue),
                doc.ingested_at.isoformat() if doc.ingested_at else None,
                now,
                now,
            ),
        )
        self.conn.commit()

    def mark_ingested(self, document_id: str, doc_hash: str, local_path: str) -> None:
        now = self._now()
        self.conn.execute(
            """
            UPDATE documents
            SET document_hash=?, local_path=?, ingested_at=?, last_seen_at=?
            WHERE document_id=?
            """,
            (doc_hash, local_path, now, now, document_id),
        )
        self.conn.commit()

    def clear_latest_flag(self, season: int, section_code: str) -> None:
        """Unset is_latest_issue for all issues of a section before setting a new one."""
        self.conn.execute(
            "UPDATE documents SET is_latest_issue=0 WHERE season=? AND section_code=?",
            (season, section_code),
        )
        self.conn.commit()

    def replace_deleted_provisions(self, document_id: str, provisions) -> None:
        self.conn.execute("DELETE FROM deleted_provisions WHERE document_id=?", (document_id,))
        self.conn.executemany(
            """
            INSERT INTO deleted_provisions
              (document_id, season, section_code, issue, page, article_id,
               appendix_id, effective_year, text)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    p.document_id, p.season, p.section_code, p.issue, p.page,
                    p.article_id, p.appendix_id, p.effective_year, p.text,
                )
                for p in provisions
            ],
        )
        self.conn.commit()

    def list_deleted_provisions(self, document_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM deleted_provisions WHERE document_id=? ORDER BY id", (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def log_ingestion_run(
        self,
        documents_new: int,
        documents_updated: int,
        documents_skipped: int,
        error: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ingestion_log (run_at, documents_new, documents_updated, documents_skipped, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self._now(), documents_new, documents_updated, documents_skipped, error),
        )
        self.conn.commit()

    def get_last_successful_check(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT run_at FROM ingestion_log WHERE error IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["run_at"] if row else None

    def get_indexed_document(self, season: int, section_code: str) -> Optional[SourceDocument]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE season=? AND section_code=? AND is_latest_issue=1 AND ingested_at IS NOT NULL",
            (season, section_code),
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def has_unresolved_in_place_update(self, document_id: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM update_journal WHERE document_id=? AND same_id=1 ORDER BY id DESC LIMIT 1",
            (document_id,),
        ).fetchone()
        return bool(row) and row["status"] in ("failed", "abandoned")

    def journal_start(self, document_id: str, season: int, section_code: str, same_id: bool, expected_chunks: int) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO update_journal (document_id, season, section_code, status, same_id, expected_chunks, started_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (document_id, season, section_code, "started", int(same_id), expected_chunks, self._now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def journal_set(self, journal_id: int, status: str, detail: Optional[str] = None) -> None:
        finished = self._now() if status in ("committed", "failed", "abandoned") else None
        self.conn.execute(
            "UPDATE update_journal SET status=?, finished_at=COALESCE(?, finished_at), detail=COALESCE(?, detail) WHERE id=?",
            (status, finished, detail, journal_id),
        )
        self.conn.commit()

    def journal_open(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM update_journal WHERE status IN ('started','indexed') ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def activate_document(
        self,
        doc: SourceDocument,
        doc_hash: str,
        local_path: str,
        provisions,
        journal_id: Optional[int] = None,
    ) -> None:
        now = self._now()
        with self.conn:
            self.conn.execute(
                "UPDATE documents SET is_latest_issue=0, status='superseded' WHERE season=? AND section_code=? AND document_id<>?",
                (doc.season, doc.section_code, doc.document_id),
            )
            self.conn.execute(
                """
                INSERT INTO documents
                  (document_id, season, section_code, section_name, issue, status, publication_date,
                   source_url, document_hash, local_path, is_latest_issue, ingested_at, last_seen_at)
                VALUES (?,?,?,?,?,'active',?,?,?,?,1,?,?)
                ON CONFLICT(document_id) DO UPDATE SET
                  section_name=excluded.section_name, issue=excluded.issue, status='active',
                  publication_date=excluded.publication_date, source_url=excluded.source_url,
                  document_hash=excluded.document_hash, local_path=excluded.local_path,
                  is_latest_issue=1, ingested_at=excluded.ingested_at, last_seen_at=excluded.last_seen_at
                """,
                (
                    doc.document_id, doc.season, doc.section_code, doc.section_name, doc.issue,
                    doc.publication_date, doc.source_url, doc_hash, local_path, now, now,
                ),
            )
            self.conn.execute("DELETE FROM deleted_provisions WHERE document_id=?", (doc.document_id,))
            self.conn.executemany(
                """
                INSERT INTO deleted_provisions
                  (document_id, season, section_code, issue, page, article_id, appendix_id, effective_year, text)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    (p.document_id, p.season, p.section_code, p.issue, p.page, p.article_id, p.appendix_id, p.effective_year, p.text)
                    for p in provisions
                ],
            )
            if journal_id is not None:
                self.conn.execute("UPDATE update_journal SET status='indexed' WHERE id=?", (journal_id,))

    def touch_document_metadata(self, document_id: str, source_url: str, publication_date: Optional[str]) -> None:
        self.conn.execute(
            "UPDATE documents SET source_url=?, publication_date=?, last_seen_at=? WHERE document_id=?",
            (source_url, publication_date, self._now(), document_id),
        )
        self.conn.commit()
