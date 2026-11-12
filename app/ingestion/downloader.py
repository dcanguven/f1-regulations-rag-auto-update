"""PDF downloader with SHA-256 hash verification."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path_for(season: int, section_code: str, issue: int) -> Path:
    return settings.raw_data_dir / str(season) / section_code / f"issue_{issue:02d}.pdf"


def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> str:
    """Download a PDF to dest and return its SHA-256 hash."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
    except httpx.HTTPError as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    file_hash = compute_sha256(dest)
    logger.info("Downloaded %s — SHA256=%s", dest.name, file_hash[:12])
    return file_hash


def is_changed(existing_hash: str | None, current_hash: str) -> bool:
    """True if the document has changed or was never downloaded."""
    return existing_hash != current_hash
