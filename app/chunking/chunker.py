from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.models.schemas import Chunk, DeletedProvision, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)

_DOC_REF_RE = re.compile(r"FIA-F1-DOC-\d+")
_ACTIVE_ROLES = ("regulatory_text", "reference_info", "unknown")

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return _tokenizer


def _word_token_counts(words: list[str]) -> list[int]:
    if not words:
        return []
    enc = _get_tokenizer()(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
    counts = [0] * len(words)
    for wid in enc.word_ids():
        if wid is not None:
            counts[wid] += 1
    return [max(c, 1) for c in counts]


def _token_count(text: str) -> int:
    return max(1, sum(_word_token_counts(text.split())))


def _make_chunk_id(document_id: str, article_id: Optional[str], index: int, scope: str = "") -> str:
    raw = f"{document_id}|{scope}|{article_id or 'none'}|{index}" if scope else f"{document_id}|{article_id or 'none'}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _ancestors(article_id: Optional[str]) -> list[str]:
    if not article_id:
        return []
    parts = article_id.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _build_embedding_text(
    season: int,
    section_name: str,
    article_id: Optional[str],
    article_title: Optional[str],
    text: str,
    article_heading: Optional[str] = None,
    appendix_id: Optional[str] = None,
    effective_year: Optional[int] = None,
    is_current: bool = True,
    is_binding: bool = True,
    effective_from: Optional[str] = None,
) -> str:
    parts = [f"{season} FIA Formula 1 Regulations", section_name]
    if appendix_id:
        parts.append(f"Appendix {appendix_id}")
    if not is_current:
        parts.append(
            f"Approved changes effective {effective_from or effective_year}"
            if effective_year
            else "Approved future changes"
        )
    if article_heading and (not article_id or "." in article_id):
        parts.append(article_heading)
    if article_id:
        label = article_id
        if article_title:
            label += f" {article_title}"
        parts.append(label)
    if not is_binding:
        parts.append("Non-binding comment, not a regulatory requirement")
    parts.append(text)
    return "\n".join(parts)


def _join_elements(elements: list[ParsedElement]) -> str:
    out: list[str] = []
    for e in elements:
        out.append(e.text)
        if e.line_end:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _is_deleted(e: ParsedElement) -> bool:
    return e.is_deleted or e.change_status == "deleted"


@dataclass
class _ArticleGroup:
    article_id: Optional[str]
    article_title: Optional[str]
    hierarchy_path: str
    elements: list[ParsedElement] = field(default_factory=list)

    @property
    def active_elements(self) -> list[ParsedElement]:
        return [e for e in self.elements if not _is_deleted(e) and e.content_role in _ACTIVE_ROLES]

    @property
    def comment_elements(self) -> list[ParsedElement]:
        return [e for e in self.elements if not _is_deleted(e) and e.content_role == "non_regulatory_comment"]

    @property
    def governance_elements(self) -> list[ParsedElement]:
        return [e for e in self.elements if not _is_deleted(e) and e.content_role == "governance_info"]

    @property
    def text(self) -> str:
        return _join_elements(self.active_elements)

    @property
    def page_start(self) -> int:
        return self.elements[0].page if self.elements else 0

    @property
    def page_end(self) -> int:
        return self.elements[-1].page if self.elements else 0

    @property
    def content_role(self):
        roles = [e.content_role for e in self.elements]
        return max(set(roles), key=roles.count)

    @property
    def token_count(self) -> int:
        return _token_count(self.text)


def _group_key(e: ParsedElement) -> tuple:
    return (e.article_id, e.hierarchy_path, e.appendix_id, e.effective_year, e.is_current)


def _group_by_article(parsed: ParsedDocument) -> list[_ArticleGroup]:
    groups: list[_ArticleGroup] = []
    current: Optional[_ArticleGroup] = None
    current_key: Optional[tuple] = None

    for elem in parsed.elements:
        if _is_deleted(elem):
            continue
        key = _group_key(elem)
        if current is None or key != current_key:
            current = _ArticleGroup(
                article_id=elem.article_id,
                article_title=elem.article_title,
                hierarchy_path=elem.hierarchy_path,
            )
            current_key = key
            groups.append(current)
        current.elements.append(elem)

    return groups


@dataclass
class _Unit:
    elements: list[ParsedElement]

    @property
    def text(self) -> str:
        return _join_elements(self.elements)

    @property
    def is_table(self) -> bool:
        return all(e.is_table for e in self.elements)


def _units(elements: list[ParsedElement]) -> list[_Unit]:
    units: list[_Unit] = []
    for e in elements:
        starts = e.item_start or e.is_table or not units or (units[-1].elements[-1].is_table != e.is_table)
        if starts:
            units.append(_Unit([e]))
        else:
            units[-1].elements.append(e)
    return [u for u in units if u.text]


def _word_pages(elements: list[ParsedElement]) -> list[tuple[str, int]]:
    words: list[tuple[str, int]] = []
    for e in elements:
        for w in e.text.split():
            words.append((w, e.page))
    return words


def _split_words(
    words: list[tuple[str, int]], max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int, int]]:
    if not words:
        return []
    counts = _word_token_counts([w for w, _ in words])
    max_tokens = max(max_tokens, 1)
    overlap_tokens = min(overlap_tokens, max_tokens // 2)
    parts: list[tuple[str, int, int]] = []
    start = 0
    n = len(words)
    while start < n:
        end = start
        used = 0
        while end < n and (used + counts[end] <= max_tokens or end == start):
            used += counts[end]
            end += 1
        seg = words[start:end]
        parts.append((" ".join(w for w, _ in seg), min(p for _, p in seg), max(p for _, p in seg)))
        if end >= n:
            break
        back = end
        kept = 0
        while back > start + 1 and kept + counts[back - 1] <= overlap_tokens:
            back -= 1
            kept += counts[back]
        start = back if back > start else end
    return parts


def _split_group(group: _ArticleGroup, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = _word_pages(group.active_elements)
    return [t for t, _, _ in _split_words(words, max_tokens, overlap_tokens)]


def _table_run_lengths(units: list[_Unit]) -> list[int]:
    runs = [0] * len(units)
    i = 0
    while i < len(units):
        if not units[i].is_table:
            i += 1
            continue
        j = i
        while j < len(units) and units[j].is_table:
            j += 1
        for k in range(i, j):
            runs[k] = j - i
        i = j
    return runs


def _row_level(run_length: int) -> bool:
    mode = settings.table_chunk_mode
    if mode == "row":
        return True
    return mode == "row_small" and run_length <= settings.table_row_chunk_max_rows


def _pack(units: list[_Unit], max_tokens: int, overlap_tokens: int) -> list[tuple[str, list[ParsedElement], bool]]:
    packed: list[tuple[str, list[ParsedElement], bool]] = []
    runs = _table_run_lengths(units)
    cur_texts: list[str] = []
    cur_elems: list[ParsedElement] = []
    cur_tokens = 0
    cur_is_table: Optional[bool] = None

    def flush() -> None:
        nonlocal cur_texts, cur_elems, cur_tokens, cur_is_table
        if cur_texts:
            packed.append(("\n".join(cur_texts), cur_elems, False))
        cur_texts, cur_elems, cur_tokens, cur_is_table = [], [], 0, None

    for index, unit in enumerate(units):
        text = unit.text
        tokens = _token_count(text)
        row_level = unit.is_table and _row_level(runs[index])
        if row_level:
            flush()
        if tokens > max_tokens:
            flush()
            for part_text, p0, p1 in _split_words(_word_pages(unit.elements), max_tokens, overlap_tokens):
                part_elems = [
                    e for e in unit.elements if p0 <= e.page <= p1
                ] or unit.elements
                packed.append((part_text, part_elems, True))
            continue
        if cur_texts and (cur_tokens + tokens > max_tokens or cur_is_table != unit.is_table):
            flush()
        cur_is_table = unit.is_table
        cur_texts.append(text)
        cur_elems.extend(unit.elements)
        cur_tokens += tokens
        if row_level:
            flush()
    flush()
    return packed


def _governance_index(groups: list[_ArticleGroup]) -> dict[tuple, str]:
    index: dict[tuple, list[str]] = {}
    for g in groups:
        gov = g.governance_elements
        if not gov:
            continue
        first = gov[0]
        top = g.article_id.split(".")[0] if g.article_id else None
        key = (first.appendix_id, top)
        text = _join_elements(gov)
        bucket = index.setdefault(key, [])
        if text not in bucket:
            bucket.append(text)
    return {k: " ".join(v) for k, v in index.items()}


def chunk_document(parsed: ParsedDocument) -> list[Chunk]:
    groups = _group_by_article(parsed)
    governance = _governance_index(groups)
    chunks: list[Chunk] = []
    chunk_index = 0
    overlap_t = settings.chunk_overlap_tokens
    seq_budget = settings.embedding_max_seq_length - 2

    for group in groups:
        if not group.elements:
            continue
        first = group.elements[0]
        top = group.article_id.split(".")[0] if group.article_id else None
        gov_text = governance.get((first.appendix_id, top))
        if gov_text is None and first.appendix_id:
            gov_text = governance.get((first.appendix_id, None))
        scope = f"{first.appendix_id}|{first.effective_year}|{group.hierarchy_path}"

        for binding, elements in ((True, group.active_elements), (False, group.comment_elements)):
            if not elements:
                continue
            header = _build_embedding_text(
                parsed.season,
                parsed.section_name,
                group.article_id,
                group.article_title,
                "",
                article_heading=first.article_heading,
                appendix_id=first.appendix_id,
                effective_year=first.effective_year,
                is_current=first.is_current,
                is_binding=binding,
            )
            max_t = max(16, min(settings.chunk_max_tokens, seq_budget - _token_count(header)))
            packed = _pack(_units(elements), max_t, overlap_t)

            for text, part_elems, was_split in packed:
                text = text.strip()
                if not text:
                    continue
                refs = sorted(
                    {
                        r
                        for e in part_elems
                        if e.content_role == "reference_info"
                        for r in _DOC_REF_RE.findall(e.text)
                    }
                    | set(_DOC_REF_RE.findall(text))
                )
                changed = any(e.change_status == "added" for e in part_elems)
                pages = [e.page for e in part_elems] or [group.page_start]
                item_path = part_elems[0].item_path if part_elems else ""
                if not binding:
                    content_type = "comment"
                elif was_split:
                    content_type = "split"
                elif part_elems and all(e.is_table for e in part_elems):
                    content_type = "table"
                else:
                    content_type = "article"
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(parsed.document_id, group.article_id, chunk_index, scope),
                        document_id=parsed.document_id,
                        season=parsed.season,
                        section_code=parsed.section_code,
                        section_name=parsed.section_name,
                        issue=parsed.issue,
                        article_id=group.article_id,
                        article_title=group.article_title,
                        hierarchy_path=group.hierarchy_path,
                        page_start=min(pages),
                        page_end=max(pages),
                        content_type=content_type,
                        content_role="regulatory_text" if binding else "non_regulatory_comment",
                        source_url=parsed.source_url,
                        text=text,
                        embedding_text=_build_embedding_text(
                            parsed.season,
                            parsed.section_name,
                            group.article_id,
                            group.article_title,
                            text,
                            article_heading=first.article_heading,
                            appendix_id=first.appendix_id,
                            effective_year=first.effective_year,
                            is_current=first.is_current,
                            is_binding=binding,
                            effective_from=first.effective_from,
                        ),
                        article_path=_ancestors(group.article_id),
                        article_heading=first.article_heading,
                        item_path=item_path,
                        appendix_id=first.appendix_id,
                        appendix_title=first.appendix_title,
                        effective_year=first.effective_year,
                        effective_from=first.effective_from,
                        is_current=first.is_current,
                        is_binding=binding,
                        contains_changes=changed,
                        changed_in_issue=parsed.issue if changed else None,
                        contains_table=any(e.is_table for e in part_elems),
                        governance=gov_text,
                        referenced_documents=refs,
                        pipeline_version=settings.pipeline_version,
                    )
                )
                chunk_index += 1

    logger.debug("Chunked %s: %d chunks from %d groups", parsed.document_id, len(chunks), len(groups))
    return chunks


def extract_deleted(parsed: ParsedDocument) -> list[DeletedProvision]:
    out: list[DeletedProvision] = []
    run: list[ParsedElement] = []

    def flush() -> None:
        if not run:
            return
        text = _join_elements(run)
        if text:
            first = run[0]
            out.append(
                DeletedProvision(
                    document_id=parsed.document_id,
                    season=parsed.season,
                    section_code=parsed.section_code,
                    issue=parsed.issue,
                    page=first.page,
                    article_id=first.article_id,
                    appendix_id=first.appendix_id,
                    effective_year=first.effective_year,
                    text=text,
                )
            )
        run.clear()

    for e in parsed.elements:
        if _is_deleted(e):
            if run and (run[0].page != e.page or run[0].article_id != e.article_id or run[0].appendix_id != e.appendix_id):
                flush()
            run.append(e)
        else:
            flush()
    flush()
    return out
