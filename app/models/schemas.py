from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


# ── Source registry ────────────────────────────────────────────────────────────

@dataclass
class SourceDocument:
    document_id: str          # fia_f1_2026_B_issue_08
    season: int
    section_code: str         # A, B, C, D, E, F
    section_name: str
    issue: int
    status: str               # active | superseded
    publication_date: Optional[str]
    source_url: str
    document_hash: Optional[str]
    local_path: Optional[str]
    is_latest_issue: bool
    ingested_at: Optional[datetime]
    last_seen_at: Optional[datetime]


# ── Parsed document ────────────────────────────────────────────────────────────

ContentRole = Literal[
    "regulatory_text",
    "governance_info",
    "reference_info",
    "non_regulatory_comment",
    "unknown",
]

ChangeStatus = Literal["unchanged", "added", "deleted"]

@dataclass
class ParsedElement:
    text: str
    page: int
    hierarchy_path: str       # "B / B5 / B5.13 / B5.13.4"
    article_id: Optional[str] # "B5.13.4"
    article_title: Optional[str]
    content_role: ContentRole
    is_deleted: bool = False
    parse_uncertain: bool = False
    font_size: Optional[float] = None
    change_status: ChangeStatus = "unchanged"
    color_hex: Optional[str] = None
    is_bold: bool = False
    item_path: str = ""
    article_heading: Optional[str] = None
    appendix_id: Optional[str] = None
    appendix_title: Optional[str] = None
    effective_year: Optional[int] = None
    effective_from: Optional[str] = None
    is_current: bool = True
    is_table: bool = False
    line_end: bool = True
    item_start: bool = False


@dataclass
class ParsedDocument:
    document_id: str
    season: int
    section_code: str
    section_name: str
    issue: int
    source_url: str
    elements: list[ParsedElement] = field(default_factory=list)
    legend: dict = field(default_factory=dict)
    front_matter_pages: int = 0


# ── Chunks ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    season: int
    section_code: str
    section_name: str
    issue: int
    article_id: Optional[str]
    article_title: Optional[str]
    hierarchy_path: str
    page_start: int
    page_end: int
    content_type: str         # "article" | "clause" | "split"
    content_role: ContentRole
    source_url: str
    text: str                 # raw text stored in Qdrant payload
    embedding_text: str       # prefixed text for embedding
    article_path: list[str] = field(default_factory=list)
    article_heading: Optional[str] = None
    item_path: str = ""
    appendix_id: Optional[str] = None
    appendix_title: Optional[str] = None
    effective_year: Optional[int] = None
    effective_from: Optional[str] = None
    is_current: bool = True
    is_binding: bool = True
    contains_changes: bool = False
    changed_in_issue: Optional[int] = None
    contains_table: bool = False
    governance: Optional[str] = None
    referenced_documents: list[str] = field(default_factory=list)
    pipeline_version: str = ""


@dataclass
class DeletedProvision:
    document_id: str
    season: int
    section_code: str
    issue: int
    page: int
    article_id: Optional[str]
    appendix_id: Optional[str]
    effective_year: Optional[int]
    text: str


# ── Retrieval ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class QueryScope:
    season: Optional[int] = None
    section_code: Optional[str] = None
    issue: Optional[int] = None
    exact_article: Optional[str] = None  # e.g. "B5.13"
    section_from_article: bool = False
    year_mentioned: Optional[int] = None
    year_mode: str = "current"


# ── API ────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    season: int
    section: str
    issue: int
    article_id: Optional[str]
    article_title: Optional[str]
    page_start: int
    page_end: int
    source_url: str
    ref: str = ""
    section_name: str = ""
    document_id: str = ""
    effective_year: Optional[int] = None
    is_current: bool = True
    is_binding: bool = True


@dataclass
class AnswerClaim:
    text: str
    sources: list[str]
    verified: bool = True
    problems: list[str] = field(default_factory=list)


@dataclass
class AskResponse:
    answer: str
    scope_used: dict
    citations: list[Citation]
    debug: Optional[dict] = None
    claims: list[AnswerClaim] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    year_interpretation: Optional[dict] = None
    context_truncated: bool = False
