"""
Query normalization, scope extraction, and exact article detection.

Handles both English and Turkish queries.
"""
from __future__ import annotations

import re
from typing import Optional

from app.config import settings
from app.models.schemas import QueryScope

# ── Normalization ──────────────────────────────────────────────────────────────
_KEYWORD = r"(?:article|art\.?|madde(?:si)?|kural[ıi]?|section|sec\.?)"

_COMPACT_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<kw>{_KEYWORD}\s+)?(?P<letter>[A-Fa-f])(?P<nums>\d{{1,3}}(?:\.\d{{1,3}})*)"
    r"(?:\.(?P<sub>[a-z]))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_SPACED_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<kw>{_KEYWORD}\s+)?(?P<letter>[A-Fa-f])\s+(?P<first>\d{{1,3}})"
    r"(?P<rest>(?:[\s.]\d{1,3})*)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_SECTION_ARTICLE_RE = re.compile(
    r"(?:section|sec\.?|b[öo]l[üu]m)\s+(?P<letter>[A-Fa-f])\b[^.?!\n]{0,40}?"
    r"\b(?:article|art\.?|madde(?:si)?)\s+(?P<nums>\d{1,3}(?:\.\d{1,3})+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_SEASON_RE = re.compile(r"\b(20[2-3]\d)\b")

_YEAR_CUE_BEFORE_RE = re.compile(
    r"\b(?:from|starting(?:\s+(?:from|in))?|beginning(?:\s+in)?|as\s+of|effective(?:\s+from)?|after|since|"
    r"changes?\s+(?:for|in|to|from)|approved\s+changes?\s+(?:for|in))\s+(?:the\s+)?"
    r"(?:(?:\d{1,2}\s+)?[A-Za-z]+\s+)?(?P<year>20[2-3]\d)\b",
    re.IGNORECASE,
)
_FUTURE_WORDS_RE = re.compile(
    r"\b(?:will|going\s+to|future|upcoming|approved\s+changes?|rule\s+changes?|new\s+rules?|next\s+(?:year|season))\b",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(
    r"(?:section|bölüm|sec\.?)\s+([A-Fa-f])\b", re.IGNORECASE
)

_ISSUE_RE = re.compile(
    r"(?:issue|iss\.?|versiyon|versiyon numarası)\s*(?:#)?\s*(\d+)", re.IGNORECASE
)


def _article_candidates(text: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for m in _COMPACT_RE.finditer(text):
        dotted = "." in m.group("nums")
        if not m.group("kw") and not (dotted and m.group("letter").isupper()):
            continue
        article = f"{m.group('letter').upper()}{m.group('nums')}"
        if m.group("sub"):
            article += f".{m.group('sub')}"
        found.append((m.start(), article))
    for m in _SPACED_RE.finditer(text):
        groups = [m.group("first")] + re.findall(r"\d{1,3}", m.group("rest"))
        if not m.group("kw") and len(groups) < 2:
            continue
        if not m.group("kw") and m.group("letter").islower() and not m.group("rest").startswith("."):
            continue
        found.append((m.start(), f"{m.group('letter').upper()}" + ".".join(groups)))
    m = _SECTION_ARTICLE_RE.search(text)
    if m:
        found.append((m.start(), f"{m.group('letter').upper()}{m.group('nums')}"))
    return sorted(found)


def normalize_article(text: str) -> Optional[str]:
    candidates = _article_candidates(text)
    return candidates[0][1] if candidates else None


_PART_SPLIT_RE = re.compile(
    r"\s*[,;]?\s+and\s+(?=(?:what|how|who|whom|when|where|which|why|can|could|is|are|does|do|did|will|may|must)\b)",
    re.IGNORECASE,
)


def split_question(question: str) -> list[str]:
    parts = [p.strip(" ,;?") for p in _PART_SPLIT_RE.split(question.strip())]
    parts = [p for p in parts if len(p.split()) >= 3]
    if 2 <= len(parts) <= 3:
        return [p + "?" for p in parts]
    return [question]


def classify_year(question: str, default_year: int) -> tuple[Optional[int], str]:
    years = [int(y) for y in _SEASON_RE.findall(question)]
    if not years:
        return None, "current"
    future = [y for y in years if y > default_year]
    if not future:
        year = years[0]
        return year, "current" if year == default_year else "past"
    year = future[0]
    cue = _YEAR_CUE_BEFORE_RE.search(question)
    if (cue and int(cue.group("year")) > default_year) or _FUTURE_WORDS_RE.search(question):
        return year, "future_explicit"
    return year, "year_mention"


def extract_scope(question: str) -> QueryScope:
    """Extract season, section, issue, and exact article from a question."""
    scope = QueryScope()

    # Season
    year, mode = classify_year(question, settings.default_season)
    scope.year_mentioned = year
    scope.year_mode = mode
    if year is not None:
        scope.season = year

    # Section
    sec_m = _SECTION_RE.search(question)
    if sec_m:
        scope.section_code = sec_m.group(1).upper()

    # Issue
    iss_m = _ISSUE_RE.search(question)
    if iss_m:
        scope.issue = int(iss_m.group(1))

    # Exact article
    article = normalize_article(question)
    if article:
        scope.exact_article = article
        if scope.section_code is None and article[0].isalpha():
            scope.section_code = article[0].upper()
            scope.section_from_article = True

    # Default season if none mentioned
    if scope.season is None:
        scope.season = settings.default_season

    return scope


def is_exact_article_query(scope: QueryScope) -> bool:
    """True when the user explicitly referenced a specific article."""
    return scope.exact_article is not None
