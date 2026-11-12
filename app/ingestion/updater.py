from __future__ import annotations

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from app.chunking.chunker import chunk_document, extract_deleted
from app.config import settings
from app.ingestion.catalog import CatalogEntry, fetch_catalog, make_document_id, pick_latest_issues
from app.ingestion.downloader import compute_sha256, download_pdf, local_path_for
from app.models.schemas import Chunk, DeletedProvision, SourceDocument
from app.parsing.pdf_parser import parse_pdf
from app.storage.index_state import bump_generation
from app.storage.qdrant_store import (
    count_document_points,
    delete_document_chunks,
    delete_superseded_documents,
    ensure_collection,
    get_qdrant_client,
    prune_obsolete_chunks,
    section_document_ids,
    upsert_chunks,
)
from app.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


MIN_CHUNK_RATIO = 0.5


class IndexIntegrityError(RuntimeError):
    pass


@dataclass
class SectionPlan:
    section_code: str
    entry: CatalogEntry
    document_id: str
    reason: str
    previous: Optional[SourceDocument] = None

    @property
    def same_id(self) -> bool:
        return self.previous is not None and self.previous.document_id == self.document_id


@dataclass
class PreparedSection:
    plan: SectionPlan
    chunks: list[Chunk]
    vectors: list[list[float]]
    provisions: list[DeletedProvision]
    document_hash: str
    part_path: Path
    final_path: Path
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class UpdateResult:
    status: str
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    planned: list[str] = field(default_factory=list)
    recovered: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "no_changes", "dry_run", "not_due") and not self.failures


@contextmanager
def update_lock() -> Iterator[bool]:
    path = Path(settings.data_dir) / ".update.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def is_due(db: SQLiteStore, now: Optional[datetime] = None) -> bool:
    last = db.get_last_successful_check()
    if not last:
        return True
    now = now or datetime.now(timezone.utc)
    age_hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600
    return age_hours >= settings.update_interval_hours


def plan_updates(
    latest: dict[str, CatalogEntry], db: SQLiteStore, force: bool = False
) -> tuple[list[SectionPlan], list[str]]:
    plans: list[SectionPlan] = []
    unchanged: list[str] = []
    for section_code, entry in sorted(latest.items()):
        doc_id = make_document_id(entry.season, section_code, entry.issue)
        active = db.get_indexed_document(entry.season, section_code)
        if active is None:
            plans.append(SectionPlan(section_code, entry, doc_id, "not_indexed"))
        elif entry.issue > active.issue:
            plans.append(SectionPlan(section_code, entry, doc_id, "new_issue", active))
        elif entry.issue < active.issue:
            logger.warning("Catalog lists %s older than indexed issue %d; ignoring", doc_id, active.issue)
            unchanged.append(active.document_id)
        elif force:
            plans.append(SectionPlan(section_code, entry, doc_id, "forced", active))
        elif db.has_unresolved_in_place_update(doc_id):
            plans.append(SectionPlan(section_code, entry, doc_id, "retry_interrupted", active))
        elif entry.source_url != active.source_url or entry.publication_date != active.publication_date:
            plans.append(SectionPlan(section_code, entry, doc_id, "metadata_changed", active))
        else:
            unchanged.append(doc_id)
    return plans, unchanged


def prepare_section(plan: SectionPlan, db: SQLiteStore, embedder: Any) -> Optional[PreparedSection]:
    entry, doc_id = plan.entry, plan.document_id
    timings: dict[str, float] = {}
    final_path = local_path_for(entry.season, plan.section_code, entry.issue)
    part_path = final_path.with_name(final_path.name + ".part")
    t = time.perf_counter()
    if plan.reason in ("new_issue", "not_indexed", "forced") and final_path.exists():
        part_path = final_path
        doc_hash = compute_sha256(final_path)
    else:
        part_path.unlink(missing_ok=True)
        doc_hash = download_pdf(entry.source_url, part_path)
    timings["download"] = time.perf_counter() - t
    with open(part_path, "rb") as handle:
        if not handle.read(5).startswith(b"%PDF"):
            if part_path != final_path:
                part_path.unlink(missing_ok=True)
            raise IndexIntegrityError(f"{doc_id}: downloaded file is not a PDF")

    if plan.reason == "metadata_changed" and plan.previous and plan.previous.document_hash == doc_hash:
        db.touch_document_metadata(doc_id, entry.source_url, entry.publication_date)
        part_path.unlink(missing_ok=True)
        logger.info("Metadata of %s changed but file content is identical; no reindex", doc_id)
        return None

    t = time.perf_counter()
    parsed = parse_pdf(
        pdf_path=part_path,
        document_id=doc_id,
        season=entry.season,
        section_code=plan.section_code,
        section_name=entry.section_name,
        issue=entry.issue,
        source_url=entry.source_url,
    )
    chunks = chunk_document(parsed)
    provisions = extract_deleted(parsed)
    timings["parse_chunk"] = time.perf_counter() - t
    if not chunks:
        raise IndexIntegrityError(f"No chunks produced for {doc_id}")

    t = time.perf_counter()
    vectors = embedder.encode([c.embedding_text for c in chunks])
    timings["embed"] = time.perf_counter() - t
    if len(vectors) != len(chunks):
        raise IndexIntegrityError(f"Embedding count mismatch for {doc_id}")
    return PreparedSection(plan, chunks, vectors, provisions, doc_hash, part_path, final_path, timings)


def commit_section(qdrant: Any, db: SQLiteStore, prepared: PreparedSection) -> int:
    plan = prepared.plan
    entry, doc_id = plan.entry, plan.document_id
    collection = settings.qdrant_collection
    previous_points = count_document_points(qdrant, collection, plan.previous.document_id) if plan.previous else 0
    if previous_points and len(prepared.chunks) < previous_points * MIN_CHUNK_RATIO:
        raise IndexIntegrityError(
            f"{doc_id}: {len(prepared.chunks)} chunks is far below the {previous_points} of the indexed issue; keeping the indexed issue"
        )
    journal_id = db.journal_start(doc_id, entry.season, plan.section_code, plan.same_id, len(prepared.chunks))
    t = time.perf_counter()
    try:
        upsert_chunks(qdrant, collection, prepared.chunks, prepared.vectors)
        if plan.same_id:
            prune_obsolete_chunks(qdrant, collection, doc_id, {c.chunk_id for c in prepared.chunks})
        indexed = count_document_points(qdrant, collection, doc_id)
        if indexed != len(prepared.chunks):
            raise IndexIntegrityError(f"{doc_id}: expected {len(prepared.chunks)} points, found {indexed}")
        doc = SourceDocument(
            document_id=doc_id,
            season=entry.season,
            section_code=plan.section_code,
            section_name=entry.section_name,
            issue=entry.issue,
            status="active",
            publication_date=entry.publication_date,
            source_url=entry.source_url,
            document_hash=prepared.document_hash,
            local_path=str(prepared.final_path),
            is_latest_issue=True,
            ingested_at=None,
            last_seen_at=None,
        )
        db.activate_document(doc, prepared.document_hash, str(prepared.final_path), prepared.provisions, journal_id)
    except BaseException as e:
        if not plan.same_id:
            try:
                delete_document_chunks(qdrant, collection, doc_id)
            except Exception:
                logger.exception("Could not roll back partial points for %s", doc_id)
        db.journal_set(journal_id, "failed", f"{type(e).__name__}: {e}")
        raise
    removed = delete_superseded_documents(qdrant, collection, entry.season, plan.section_code, doc_id)
    db.journal_set(journal_id, "committed")
    if prepared.part_path != prepared.final_path:
        os.replace(prepared.part_path, prepared.final_path)
    logger.info(
        "Committed %s: %d chunks, %d superseded vectors removed, %.2fs",
        doc_id, len(prepared.chunks), removed, time.perf_counter() - t,
    )
    return removed


def recover(qdrant: Any, db: SQLiteStore) -> int:
    collection = settings.qdrant_collection
    actions = 0
    for entry in db.journal_open():
        doc_id, season, section = entry["document_id"], entry["season"], entry["section_code"]
        if entry["status"] == "started":
            if not entry["same_id"]:
                delete_document_chunks(qdrant, collection, doc_id)
            db.journal_set(entry["id"], "abandoned", "interrupted before activation")
            logger.warning("Recovered interrupted update of %s: partial points removed, previous issue kept", doc_id)
        else:
            removed = delete_superseded_documents(qdrant, collection, season, section, doc_id)
            db.journal_set(entry["id"], "committed", "completed after interruption")
            logger.warning("Recovered interrupted update of %s: %d superseded vectors removed", doc_id, removed)
        actions += 1
    sections = {(d.season, d.section_code) for d in db.list_documents(latest_only=True)}
    for season, section in sorted(sections):
        active = db.get_indexed_document(season, section)
        if not active:
            continue
        present = section_document_ids(qdrant, collection, season, section)
        if active.document_id in present and present - {active.document_id}:
            removed = delete_superseded_documents(qdrant, collection, season, section, active.document_id)
            logger.warning("Removed %d stale vectors of %s", removed, sorted(present - {active.document_id}))
            actions += 1
    return actions


def run_update(
    force: bool = False,
    if_due: bool = False,
    dry_run: bool = False,
    embedder: Any = None,
    fetch: Optional[Callable[[], list[CatalogEntry]]] = None,
) -> UpdateResult:
    t0 = time.perf_counter()
    result = UpdateResult(status="ok")
    with update_lock() as acquired:
        if not acquired:
            logger.info("Another update is already running; exiting")
            result.status = "busy"
            return result
        with SQLiteStore(settings.db_file) as db:
            if if_due and not force and not db.journal_open() and not is_due(db):
                logger.info("Last successful check is recent (< %.0fh); nothing to do", settings.update_interval_hours)
                result.status = "not_due"
                return result
            logger.info("Update check started (force=%s dry_run=%s)", force, dry_run)
            try:
                entries = (fetch or fetch_catalog)()
            except RuntimeError as e:
                logger.error("Catalog fetch failed: %s; existing index preserved", e)
                db.log_ingestion_run(0, 0, 0, error=str(e))
                result.status = "failed"
                result.failures["catalog"] = str(e)
                result.seconds = time.perf_counter() - t0
                return result
            latest = pick_latest_issues(entries)
            if not latest:
                message = "Catalog returned no 2026 A-F documents"
                logger.error("%s; existing index preserved", message)
                db.log_ingestion_run(0, 0, 0, error=message)
                result.status = "failed"
                result.failures["catalog"] = message
                result.seconds = time.perf_counter() - t0
                return result
            logger.info("Catalog latest issues: %s", {k: v.issue for k, v in sorted(latest.items())})
            plans, result.unchanged = plan_updates(latest, db, force)
            result.planned = [p.document_id for p in plans]
            pending_recovery = bool(db.journal_open())
            if dry_run:
                result.status = "dry_run"
                logger.info("Dry run: would update %s", result.planned or "nothing")
                return result
            if not plans and not pending_recovery:
                db.log_ingestion_run(0, 0, len(result.unchanged))
                result.status = "no_changes"
                result.seconds = time.perf_counter() - t0
                logger.info("No changes in %.1fs; index untouched", result.seconds)
                return result

            prepared: list[PreparedSection] = []
            if plans:
                if embedder is None:
                    from app.retrieval.embedder import Embedder
                    embedder = Embedder()
                for plan in plans:
                    logger.info("Processing %s (%s)", plan.document_id, plan.reason)
                    try:
                        item = prepare_section(plan, db, embedder)
                    except Exception as e:
                        logger.exception("Processing failed for %s", plan.document_id)
                        result.failures[plan.document_id] = f"{type(e).__name__}: {e}"
                        plan_part = local_path_for(plan.entry.season, plan.section_code, plan.entry.issue)
                        plan_part.with_name(plan_part.name + ".part").unlink(missing_ok=True)
                        continue
                    if item is None:
                        result.unchanged.append(plan.document_id)
                        continue
                    logger.info(
                        "Prepared %s: %d chunks (%s)", plan.document_id, len(item.chunks),
                        ", ".join(f"{k}={v:.1f}s" for k, v in item.timings.items()),
                    )
                    prepared.append(item)

            if prepared or pending_recovery:
                qdrant = get_qdrant_client()
                try:
                    ensure_collection(qdrant, settings.qdrant_collection, settings.embedding_dim)
                    result.recovered = recover(qdrant, db)
                    for item in prepared:
                        try:
                            commit_section(qdrant, db, item)
                        except Exception as e:
                            logger.exception("Commit failed for %s; previous index kept", item.plan.document_id)
                            result.failures[item.plan.document_id] = f"{type(e).__name__}: {e}"
                            continue
                        (result.updated if item.plan.same_id else result.new).append(item.plan.document_id)
                    if result.new or result.updated or result.recovered:
                        bump_generation()
                finally:
                    qdrant.close()

            error = "; ".join(f"{k}: {v}" for k, v in result.failures.items()) or None
            db.log_ingestion_run(len(result.new), len(result.updated), len(result.unchanged), error=error)
            result.status = "failed" if result.failures else "ok"
            result.seconds = time.perf_counter() - t0
            logger.info(
                "Update finished in %.1fs: status=%s new=%s updated=%s unchanged=%d failures=%s",
                result.seconds, result.status, result.new, result.updated, len(result.unchanged), list(result.failures),
            )
    return result
