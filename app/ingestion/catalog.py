"""FIA catalog discovery — scrape the HTML catalog page for 2026 A-F PDF links."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

# Section codes we target
TARGET_SECTIONS = {"A", "B", "C", "D", "E", "F"}

SECTION_NAMES = {
    "A": "General Provisions",
    "B": "Sporting",
    "C": "Technical",
    "D": "Financial - F1 Teams",
    "E": "Financial - PU Manufacturers",
    "F": "Operational",
}

# Regex to parse filenames like:
#   fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf
#   fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf
_FILENAME_RE = re.compile(
    r"fia_(?P<season>\d{4})_f1_regulations"
    r".*?section_(?P<section>[a-f])"
    r".*?iss[_\s]?(?P<issue>\d+)"
    r".*?(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


@dataclass
class CatalogEntry:
    season: int
    section_code: str          # uppercase A-F
    section_name: str
    issue: int
    publication_date: str      # YYYY-MM-DD
    source_url: str            # absolute URL


def fetch_catalog(timeout: float = 30.0) -> list[CatalogEntry]:
    """Download the FIA regulation catalog page and extract all 2026 A-F entries."""
    try:
        resp = httpx.get(settings.fia_catalog_url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch FIA catalog: {e}") from e

    return _parse_html(resp.text)


def _parse_html(html: str) -> list[CatalogEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[CatalogEntry] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if ".pdf" not in href.lower():
            continue

        url = href if href.startswith("http") else settings.fia_base_url + href
        filename = url.split("/")[-1]

        m = _FILENAME_RE.search(filename)
        if not m:
            continue

        season = int(m.group("season"))
        section = m.group("section").upper()
        issue = int(m.group("issue"))
        date = m.group("date")

        if season != settings.default_season:
            continue
        if section not in TARGET_SECTIONS:
            continue

        entries.append(
            CatalogEntry(
                season=season,
                section_code=section,
                section_name=SECTION_NAMES.get(section, section),
                issue=issue,
                publication_date=date,
                source_url=url,
            )
        )

    logger.info("Discovered %d 2026 A-F entries from catalog", len(entries))
    return entries


def pick_latest_issues(entries: list[CatalogEntry]) -> dict[str, CatalogEntry]:
    """Return the highest-issue entry for each section code."""
    latest: dict[str, CatalogEntry] = {}
    for e in entries:
        key = e.section_code
        if key not in latest or e.issue > latest[key].issue:
            latest[key] = e
    return latest


def make_document_id(season: int, section_code: str, issue: int) -> str:
    return f"fia_f1_{season}_{section_code}_issue_{issue:02d}"
