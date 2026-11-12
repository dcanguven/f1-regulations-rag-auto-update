"""
Synchronisation entry point.

Usage:
    python -m app.ingestion.sync [--if-due] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import settings
from app.ingestion.updater import UpdateResult, run_update

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_dir = Path(settings.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_dir / "update.log", maxBytes=1_000_000, backupCount=5))
    except OSError:
        pass
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def run_sync(force: bool = False, if_due: bool = False, dry_run: bool = False) -> UpdateResult:
    return run_update(force=force, if_due=if_due, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the FIA catalog and update the index safely")
    parser.add_argument("--if-due", action="store_true", help="skip when the last successful check is recent")
    parser.add_argument("--force", action="store_true", help="reprocess the latest issues even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="report planned updates without changing anything")
    args = parser.parse_args(argv)
    configure_logging()
    result = run_sync(force=args.force, if_due=args.if_due, dry_run=args.dry_run)
    return 0 if result.ok or result.status == "busy" else 1


if __name__ == "__main__":
    sys.exit(main())
