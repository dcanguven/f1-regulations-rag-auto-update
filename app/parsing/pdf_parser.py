from __future__ import annotations

import colorsys
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf as fitz

from app.models.schemas import ContentRole, ParsedDocument, ParsedElement
from app.parsing.text_repair import clean_text

logger = logging.getLogger(__name__)

HEADING_X_MAX = 60.0
TITLE_X_MIN = 85.0
FRONT_MATTER_MIN_DIGIT_LINES = 8
TOP_BAND = 0.05
BOTTOM_BAND = 0.93
FOOTER_BAND = 0.90
TABLE_MERGE_MAX_GAP = 12.0
TABLE_MERGE_EDGE_TOLERANCE = 6.0
MAX_HEADER_ROWS = 4
CAPTION_MIN_WIDTH_RATIO = 0.5
STRIKE_MAX_HEIGHT = 1.3
STRIKE_MIN_WIDTH = 2.0
STRIKE_VERTICAL_TOLERANCE = 0.22
ITEM_COLUMNS = ((88.0, 97.0, 1), (108.0, 119.0, 2), (129.0, 141.0, 3))

_ARTICLE_RE = re.compile(
    r"^(?P<article>[A-F]\d{1,3}(?:\.\d{1,3})*(?:\.[a-z])?)\s*(?P<title>.*)$"
)
_ID_LINE_RE = re.compile(r"^(?P<id>[A-F]\d{1,3}(?:\.\d{1,3})*)(?:\s+(?P<title>\S.*))?$")
_ARTICLE_HEAD_RE = re.compile(r"^ARTICLE\s+(?P<id>[A-F]\d{1,3})\s*:\s*(?P<title>\S.*)$")
_APPENDIX_HEAD_RE = re.compile(r"^APPENDIX\s+(?P<id>[A-F]\d{1,3})\s*:\s*(?P<title>\S.*)$")
_CHANGES_FOR_RE = re.compile(r"^Changes\s+for\s+(?P<year>20\d{2})\b")
_PERIOD_CAPTION_RE = re.compile(
    r"^Approved\s+changes\s+for\b.*?\b(?:starting|commencing|beginning)\s+(?:from\s+)?"
    r"(?:(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+)?(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_NUMERIC_CELL_RE = re.compile(r"^[\d\s’',.\-−+%]*\d[\d\s’',.\-−+%]*$")
TOC_MIN_DIGIT_LINES = 3
TOC_MIN_DIGIT_RATIO = 0.15
HEADER_CLUSTER_TOLERANCE = 18.0
HEADER_MAX_CHARS = 60
HEADER_MAX_WORDS = 10
_ITEM_RE = re.compile(r"^(?P<label>[a-z]|[ivxl]{1,6})\.(?:\s+(?P<rest>\S.*))?$")
_LEGEND_WORDS = {"pink": "added", "red": "governance", "orange": "reference", "green": "comment"}
_ROLE_BY_CLASS: dict[str, ContentRole] = {
    "normal": "regulatory_text",
    "added": "regulatory_text",
    "governance": "governance_info",
    "reference": "reference_info",
    "comment": "non_regulatory_comment",
}
_NAVY = {0x002D5F, 0x002060}
_FOOTER_TEXT_RE = re.compile(
    r"^(?:\d{1,2}\s+[A-Za-z]+\s+20\d{2}|20\d{2}\s+(?:fia\s+)?formula\s+(?:1|one)\s+regulations\b.*|©\s?\d{4}.*)$",
    re.IGNORECASE,
)
_PAGE_TAG_RE = re.compile(r"^[A-F]\d{1,3}$")
_WHITE = 0xFFFFFF
LEGEND_TOLERANCE = 48.0


def _rgb_to_hex(rgb: tuple) -> str:
    r, g, b = (int(round(x * 255)) for x in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _int_to_rgb(color: int) -> tuple[int, int, int]:
    return (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF


def _int_to_hex(color: int) -> str:
    r, g, b = _int_to_rgb(color)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_int(color_hex: str) -> int:
    return int(color_hex.lstrip("#"), 16)


def _hue_family(color: int) -> str:
    r, g, b = _int_to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hue = h * 360
    if s < 0.45 or v < 0.55:
        return "normal"
    if 270 <= hue <= 345:
        return "added"
    if hue >= 345 or hue <= 12:
        return "governance"
    if 20 <= hue <= 52:
        return "reference"
    if 90 <= hue <= 165:
        return "comment"
    return "normal"


def classify_colour(color: int, legend: Optional[dict[str, list[int]]] = None) -> str:
    if legend:
        r, g, b = _int_to_rgb(color)
        best: Optional[tuple[float, str]] = None
        for cls, swatches in legend.items():
            for sw in swatches:
                sr, sg, sb = _int_to_rgb(sw)
                dist = ((r - sr) ** 2 + (g - sg) ** 2 + (b - sb) ** 2) ** 0.5
                if dist <= LEGEND_TOLERANCE and (best is None or dist < best[0]):
                    best = (dist, cls)
        if best:
            return best[1]
    return _hue_family(color)


def _classify_role(color_hex: str, font_flags: int = 0, font_size: float = 10.0) -> ContentRole:
    return _ROLE_BY_CLASS[classify_colour(_hex_to_int(color_hex))]


def _is_header_footer(
    text: str,
    page_num: int,
    page_height: float,
    y: float,
    color: Optional[int] = None,
) -> bool:
    if y < page_height * TOP_BAND:
        return True
    if y > page_height * BOTTOM_BAND and (color is None or color in _NAVY or color == _WHITE):
        return True
    stripped = text.strip()
    if y > page_height * FOOTER_BAND:
        if _FOOTER_TEXT_RE.match(stripped):
            return True
        if color == _WHITE and _PAGE_TAG_RE.match(stripped):
            return True
    if re.match(r"^(?:issue|iss)\s*\d", stripped, re.IGNORECASE):
        return True
    if re.match(r"^\d{4}\s+fia\s+formula", stripped, re.IGNORECASE):
        return True
    if re.match(r"^© ?\d{4}", stripped):
        return True
    if re.match(r"^page\s+\d+", stripped, re.IGNORECASE):
        return True
    return False


def _extract_article_id(text: str) -> tuple[Optional[str], Optional[str]]:
    stripped = text.strip()
    m = _ARTICLE_RE.match(stripped)
    if m:
        return m.group("article"), m.group("title").strip() or None
    return None, None


def _ancestors(article_id: str) -> list[str]:
    parts = article_id.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _future_caption(text: str) -> Optional[tuple[int, Optional[str]]]:
    m = _CHANGES_FOR_RE.match(text)
    if m:
        return int(m.group("year")), None
    m = _PERIOD_CAPTION_RE.match(text)
    if m:
        effective_from = (
            f"{m.group('day')} {m.group('month')} {m.group('year')}" if m.group("day") else None
        )
        return int(m.group("year")), effective_from
    return None


class HierarchyTracker:
    def __init__(self, section_code: str, season: Optional[int] = None) -> None:
        self.section_code = section_code
        self.season = season
        self.current_article: Optional[str] = None
        self.current_article_title: Optional[str] = None
        self.appendix_id: Optional[str] = None
        self.appendix_title: Optional[str] = None
        self._future = False
        self._effective_year: Optional[int] = None
        self.effective_from: Optional[str] = None
        self.target_appendix: Optional[str] = None
        self._titles: dict[str, str] = {}
        self._headings: dict[str, str] = {}
        self._path_parts: list[str] = [section_code]

    def push(self, article_id: str, title: Optional[str]) -> None:
        self.current_article = article_id
        if title:
            self._titles[article_id] = title
        self.current_article_title = self._lookup_title(article_id)
        if "." not in article_id and title:
            self._headings[article_id] = f"{article_id} {title}"
        self._rebuild_path()

    def _lookup_title(self, article_id: str) -> Optional[str]:
        for anc in reversed(_ancestors(article_id)):
            if anc in self._titles:
                return self._titles[anc]
        return None

    def set_title(self, article_id: str, title: str) -> None:
        self._titles[article_id] = title
        if self.current_article == article_id:
            self.current_article_title = title

    def _rebuild_path(self) -> None:
        parts = [self.section_code]
        if self.appendix_id:
            parts.append(f"APPENDIX {self.appendix_id}")
            if self._future and self._effective_year:
                parts.append(f"Changes for {self._effective_year}")
            if self._future and self.target_appendix:
                parts.append(f"APPENDIX {self.target_appendix}")
        if self.current_article:
            parts.extend(_ancestors(self.current_article))
        self._path_parts = parts

    def enter_appendix(self, appendix_id: str, title: str) -> None:
        self.appendix_id = appendix_id
        self.appendix_title = title
        self._future = "SUBSEQUENT YEARS" in title.upper()
        self._effective_year = None
        self.effective_from = None
        self.target_appendix = None
        self.current_article = None
        self.current_article_title = None
        self._rebuild_path()

    def enter_target_appendix(self, appendix_id: str) -> None:
        self.target_appendix = appendix_id
        self.current_article = None
        self.current_article_title = None
        self._rebuild_path()

    def set_effective_year(self, year: int, effective_from: Optional[str] = None) -> None:
        self._effective_year = year
        self.effective_from = effective_from
        self.target_appendix = None
        self.current_article = None
        self.current_article_title = None
        self._rebuild_path()

    @property
    def in_future_appendix(self) -> bool:
        return self._future

    @property
    def is_current(self) -> bool:
        return not self._future

    @property
    def effective_year(self) -> Optional[int]:
        if self._future:
            return self._effective_year
        return self.season

    @property
    def article_heading(self) -> Optional[str]:
        if not self.current_article:
            return None
        return self._headings.get(self.current_article.split(".")[0])

    def path(self) -> str:
        return " / ".join(self._path_parts)

    def path_with_sub(self, sub: str) -> str:
        return " / ".join(self._path_parts + [sub])


@dataclass
class _Seg:
    text: str
    cls: str
    struck: bool
    color: int
    size: float


@dataclass
class _Line:
    page: int
    x0: float
    y0: float
    y1: float
    segs: list[_Seg]
    first_bold: bool
    first_color: int
    size: float
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segs)

    @property
    def first_struck(self) -> bool:
        return self.segs[0].struck if self.segs else False


def _is_bold(span: dict) -> bool:
    return "bold" in span.get("font", "").lower() or bool(span.get("flags", 0) & 16)


def _read_legend(page) -> dict[str, list[int]]:
    legend: dict[str, list[int]] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            joined = " ".join(s["text"] for s in spans).lower()
            for word, cls in _LEGEND_WORDS.items():
                if f"{word} text" not in joined:
                    continue
                for s in spans:
                    if word in s["text"].lower() and s["color"] != 0:
                        legend.setdefault(cls, []).append(s["color"])
                        break
    return legend


def _count_front_matter_pages(pdf) -> int:
    count = 0
    for page in pdf:
        height = page.rect.height
        digits = 0
        total = 0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                y = line["bbox"][1]
                if not text or not (height * TOP_BAND < y < height * BOTTOM_BAND):
                    continue
                total += 1
                if re.fullmatch(r"\d{1,3}", text) and line["bbox"][0] > 480:
                    digits += 1
        if digits >= FRONT_MATTER_MIN_DIGIT_LINES or (
            digits >= TOC_MIN_DIGIT_LINES and total and digits / total >= TOC_MIN_DIGIT_RATIO
        ):
            count += 1
        else:
            break
    return max(count, 1)


def _strike_rects(page) -> list[tuple[float, float, float]]:
    rects: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or _hue_family(_hex_to_int(_rgb_to_hex(fill))) != "added":
            continue
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            r = item[1]
            if r.height < STRIKE_MAX_HEIGHT and r.width > STRIKE_MIN_WIDTH:
                rects.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    return rects


def _is_struck(bbox: tuple, rects: list[tuple[float, float, float]]) -> bool:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    mid = (y0 + y1) / 2
    height = y1 - y0
    for rx0, rx1, ry in rects:
        if rx0 - 0.6 <= cx <= rx1 + 0.6 and abs(ry - mid) < STRIKE_VERTICAL_TOLERANCE * height:
            return True
    return False


def _read_lines(page, page_num: int, legend, rects) -> list[_Line]:
    data = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    lines: list[_Line] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            chars: list[list] = []
            first_bold: Optional[bool] = None
            for span in line["spans"]:
                color = span["color"]
                cls = classify_colour(color, legend)
                bold = _is_bold(span)
                for c in span["chars"]:
                    ch = c["c"]
                    struck = cls == "added" and not ch.isspace() and _is_struck(c["bbox"], rects)
                    chars.append([ch, cls, struck, color, span["size"]])
                    if first_bold is None and not ch.isspace():
                        first_bold = bold
            if first_bold is None:
                continue
            nxt = None
            for entry in reversed(chars):
                if entry[0].isspace():
                    if nxt is not None:
                        entry[1:] = nxt[1:]
                else:
                    nxt = entry
            segs: list[_Seg] = []
            for ch, cls, struck, color, size in chars:
                if segs and segs[-1].cls == cls and segs[-1].struck == struck:
                    segs[-1].text += ch
                else:
                    segs.append(_Seg(ch, cls, struck, color, size))
            segs[0].text = segs[0].text.lstrip()
            segs[-1].text = segs[-1].text.rstrip()
            segs = [s for s in segs if s.text]
            if not segs:
                continue
            first = next(s for s in segs if s.text.strip())
            bbox = line["bbox"]
            lines.append(
                _Line(
                    page=page_num,
                    x0=bbox[0],
                    y0=bbox[1],
                    y1=bbox[3],
                    segs=segs,
                    first_bold=bool(first_bold),
                    first_color=first.color,
                    size=first.size,
                    bbox=tuple(bbox),
                )
            )
    return lines


def _heading_kind(line: _Line, section_code: str) -> Optional[tuple[str, str, Optional[str]]]:
    if not line.first_bold or line.x0 >= HEADING_X_MAX or line.first_struck:
        return None
    text = clean_text(line.text).strip()
    m = _ARTICLE_HEAD_RE.match(text)
    if m and m.group("id").startswith(section_code):
        return "article", m.group("id"), m.group("title").strip()
    m = _APPENDIX_HEAD_RE.match(text)
    if m and m.group("id").startswith(section_code):
        return "appendix", m.group("id"), m.group("title").strip()
    m = _ID_LINE_RE.match(text)
    if m and m.group("id").startswith(section_code):
        title = m.group("title")
        return "clause", m.group("id"), title.strip() if title else None
    return None


def _item_level(line: _Line) -> Optional[tuple[int, str, bool]]:
    if line.first_bold:
        return None
    m = _ITEM_RE.match(clean_text(line.text).strip())
    if not m:
        return None
    for lo, hi, level in ITEM_COLUMNS:
        if lo <= line.x0 < hi:
            return level, m.group("label"), bool(m.group("rest"))
    return None


@dataclass
class _Table:
    bbox: tuple
    rows: list[list[Optional[tuple]]]
    labels: list[str] = field(default_factory=list)


def _has_numeric_cell(rows: list[list]) -> bool:
    return any(c and _NUMERIC_CELL_RE.match(norm_cell(c)) for row in rows for c in row)


def norm_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _merge_split_tables(found: list[tuple[_Table, list[list]]]) -> list[_Table]:
    merged: list[tuple[_Table, list[list]]] = []
    for table, extract in sorted(found, key=lambda x: x[0].bbox[1]):
        if merged:
            prev, prev_extract = merged[-1]
            gap = table.bbox[1] - prev.bbox[3]
            if (
                -3.0 <= gap <= TABLE_MERGE_MAX_GAP
                and abs(prev.bbox[2] - table.bbox[2]) <= TABLE_MERGE_EDGE_TOLERANCE
                and prev.bbox[0] >= table.bbox[0] - TABLE_MERGE_EDGE_TOLERANCE
                and not _has_numeric_cell(prev_extract)
                and _has_numeric_cell(extract)
            ):
                union = (
                    min(prev.bbox[0], table.bbox[0]),
                    prev.bbox[1],
                    max(prev.bbox[2], table.bbox[2]),
                    max(prev.bbox[3], table.bbox[3]),
                )
                merged[-1] = (_Table(bbox=union, rows=prev.rows + table.rows), prev_extract + extract)
                continue
        merged.append((table, extract))
    return [t for t, _ in merged]


def _page_tables(page) -> list[_Table]:
    try:
        found = page.find_tables()
    except Exception:
        return []
    items: list[tuple[_Table, list[list]]] = []
    for t in found.tables:
        rows = [list(row.cells) for row in t.rows]
        if len(rows) >= 1 and any(c is not None for r in rows for c in r):
            items.append((_Table(bbox=tuple(t.bbox), rows=rows), t.extract()))
    return _merge_split_tables(items)


def _in_bbox(x: float, y: float, bbox: tuple) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _cell_content(lines: list[_Line]) -> tuple[str, str, bool, bool, Optional[float]]:
    kept: list[str] = []
    deleted: list[str] = []
    centers: list[float] = []
    added = False
    comment_only = True
    for ln in sorted(lines, key=lambda x: (round(x.y0), x.x0)):
        centers.append((ln.bbox[0] + ln.bbox[2]) / 2)
        for s in ln.segs:
            if s.struck:
                deleted.append(s.text)
                continue
            kept.append(s.text)
            if s.cls == "added":
                added = True
            if s.cls != "comment":
                comment_only = False
        kept.append(" ")
    text = re.sub(r"\s+", " ", clean_text("".join(kept))).strip()
    xc = sum(centers) / len(centers) if centers else None
    return (
        text,
        re.sub(r"\s+", " ", clean_text(" ".join(deleted))).strip(),
        added,
        comment_only and bool(text),
        xc,
    )


def _looks_like_header(cells: list[str]) -> bool:
    texts = [c for c in cells if c]
    if not texts:
        return False
    return all(len(c) <= HEADER_MAX_CHARS and len(c.split()) <= HEADER_MAX_WORDS for c in texts)


def _center_x(bbox: Optional[tuple]) -> Optional[float]:
    return None if bbox is None else (bbox[0] + bbox[2]) / 2


def _labels_for(
    header_cells: list[list[tuple[str, tuple]]], cx: Optional[float], used: set
) -> str:
    if cx is None:
        return ""
    parts: list[str] = []
    for hi, row in enumerate(header_cells):
        for hj, (text, bbox) in enumerate(row):
            if bbox[0] <= cx <= bbox[2]:
                used.add((hi, hj))
                if not parts or parts[-1] != text:
                    parts.append(text)
    return " ".join(parts)


def _table_elements(table: _Table, cell_lines: dict, tracker: HierarchyTracker, page: int, table_index: int):
    grid: list[list[str]] = []
    row_meta: list[tuple[str, bool, bool]] = []
    for r, row in enumerate(table.rows):
        texts: list[str] = []
        deleted_parts: list[str] = []
        added_any = False
        comment_all = True
        has_text = False
        for c, cell in enumerate(row):
            if cell is None:
                texts.append("")
                continue
            text, deleted, added, comment_only, _ = _cell_content(cell_lines.get((table_index, r, c), []))
            texts.append(text)
            if deleted:
                deleted_parts.append(deleted)
            if text:
                has_text = True
                added_any = added_any or added
                comment_all = comment_all and comment_only
        grid.append(texts)
        row_meta.append((" ".join(deleted_parts), added_any, has_text and comment_all))
    filled = [i for i, row in enumerate(grid) if any(row)]
    ncols = max((len(r) for r in grid), default=0)
    out: list[ParsedElement] = []

    def make(text: str, status: str, added: bool, comment: bool) -> ParsedElement:
        return ParsedElement(
            text=text,
            page=page,
            hierarchy_path=tracker.path(),
            article_id=tracker.current_article,
            article_title=tracker.current_article_title,
            content_role="non_regulatory_comment" if comment else "regulatory_text",
            is_deleted=status == "deleted",
            change_status="deleted" if status == "deleted" else ("added" if added else "unchanged"),
            appendix_id=tracker.appendix_id,
            appendix_title=tracker.appendix_title,
            effective_year=tracker.effective_year,
            effective_from=tracker.effective_from,
            is_current=tracker.is_current,
            article_heading=tracker.article_heading,
            is_table=True,
            item_start=True,
        )

    emitted_deleted: set[int] = set()

    def emit_deleted(i: int) -> None:
        if i not in emitted_deleted and row_meta[i][0]:
            emitted_deleted.add(i)
            out.append(make(row_meta[i][0], "deleted", False, False))

    def finish() -> list[ParsedElement]:
        for i in range(len(grid)):
            emit_deleted(i)
        return out

    if not filled:
        return finish()

    def emit(i: int, parts: list[str]) -> None:
        _, added, comment = row_meta[i]
        if parts:
            out.append(make(" | ".join(parts), "kept", added, comment))
        emit_deleted(i)

    table_width = max(table.bbox[2] - table.bbox[0], 1.0)

    def is_caption(i: int) -> bool:
        filled_cells = [j for j, c in enumerate(grid[i]) if c]
        if len(filled_cells) != 1 or ncols < 2:
            return False
        bbox = table.rows[i][filled_cells[0]]
        return bbox is not None and (bbox[2] - bbox[0]) >= CAPTION_MIN_WIDTH_RATIO * table_width

    def has_numeric(i: int) -> bool:
        return any(_NUMERIC_CELL_RE.match(c) for c in grid[i] if c)

    pos = 0
    captions: list[str] = []
    while pos < len(filled) and is_caption(filled[pos]) and pos < len(filled) - 1:
        captions.append(next(c for c in grid[filled[pos]] if c))
        pos += 1
    rest = filled[pos:]
    any_numeric = any(has_numeric(i) for i in rest)
    header_rows: list[int] = []
    for idx, i in enumerate(rest):
        if len(header_rows) >= MAX_HEADER_ROWS or is_caption(i) or not _looks_like_header(grid[i]):
            break
        if has_numeric(i) and (captions or idx > 0):
            break
        header_rows.append(i)
    if not any_numeric:
        header_rows = header_rows[:1] if len(rest) > 1 else []
    data_rows = rest[len(header_rows):]
    header_cells = [
        [(grid[i][j], table.rows[i][j]) for j in range(len(grid[i])) if grid[i][j] and table.rows[i][j] is not None]
        for i in header_rows
    ]
    labelled = any(header_cells)
    prefix = " ".join(captions)

    if not data_rows:
        emit(rest[0] if rest else filled[-1], [c for c in grid[rest[0] if rest else filled[-1]] if c])
        return finish()

    used_headers: set = set()
    pending_caption: Optional[tuple[int, str]] = None
    for i in data_rows:
        if is_caption(i) and i != data_rows[-1]:
            if pending_caption is not None:
                emit(pending_caption[0], [pending_caption[1]])
            pending_caption = (i, next(c for c in grid[i] if c))
            continue
        row_prefix = " ".join(x for x in (prefix, pending_caption[1] if pending_caption else "") if x)
        parts: list[str] = []
        for j, cell in enumerate(grid[i]):
            if not cell:
                continue
            label = _labels_for(header_cells, _center_x(table.rows[i][j]), used_headers) if labelled else ""
            parts.append(f"{label}: {cell}" if label and label != cell else cell)
        if parts and row_prefix:
            if labelled:
                parts.insert(0, row_prefix.rstrip(":"))
            else:
                parts[0] = f"{row_prefix} {parts[0]}"
        emit(i, parts)
    unused = [
        text
        for hi, row in enumerate(header_cells)
        for hj, (text, _) in enumerate(row)
        if (hi, hj) not in used_headers
    ]
    if unused:
        lead = [prefix.rstrip(":")] if prefix else []
        emit(header_rows[0], lead + unused)
    return finish()


def _make_elements(line: _Line, tracker: HierarchyTracker, item_path: str, item_start: bool) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    last = len(line.segs) - 1
    for i, seg in enumerate(line.segs):
        text = clean_text(seg.text)
        if not text:
            continue
        status = "deleted" if seg.struck else ("added" if seg.cls == "added" else "unchanged")
        elements.append(
            ParsedElement(
                text=text,
                page=line.page,
                hierarchy_path=tracker.path(),
                article_id=tracker.current_article,
                article_title=tracker.current_article_title,
                content_role=_ROLE_BY_CLASS[seg.cls],
                is_deleted=seg.struck,
                font_size=seg.size,
                change_status=status,
                color_hex=_int_to_hex(seg.color),
                is_bold=line.first_bold and i == 0,
                item_path=item_path,
                article_heading=tracker.article_heading,
                appendix_id=tracker.appendix_id,
                appendix_title=tracker.appendix_title,
                effective_year=tracker.effective_year,
                effective_from=tracker.effective_from,
                is_current=tracker.is_current,
                line_end=i == last,
                item_start=item_start and i == 0,
            )
        )
    return elements


def parse_pdf(
    pdf_path: Path | str,
    document_id: str,
    season: int,
    section_code: str,
    section_name: str,
    issue: int,
    source_url: str,
) -> ParsedDocument:
    doc = ParsedDocument(
        document_id=document_id,
        season=season,
        section_code=section_code,
        section_name=section_name,
        issue=issue,
        source_url=source_url,
    )

    pdf = fitz.open(str(pdf_path))
    legend = _read_legend(pdf[0])
    front_pages = _count_front_matter_pages(pdf)
    doc.legend = {cls: [_int_to_hex(c) for c in cols] for cls, cols in legend.items()}
    doc.front_matter_pages = front_pages
    tracker = HierarchyTracker(section_code, season)
    item_stack: list[str] = []

    for page_num in range(front_pages + 1, pdf.page_count + 1):
        page = pdf[page_num - 1]
        page_height = page.rect.height
        rects = _strike_rects(page)
        lines = _read_lines(page, page_num, legend, rects)
        tables = _page_tables(page)

        membership: dict[int, tuple[int, int, int]] = {}
        cell_lines: dict[tuple[int, int, int], list[_Line]] = {}
        if tables:
            for li, ln in enumerate(lines):
                cx = (ln.bbox[0] + ln.bbox[2]) / 2
                cy = (ln.bbox[1] + ln.bbox[3]) / 2
                for ti, tb in enumerate(tables):
                    if not _in_bbox(cx, cy, tb.bbox):
                        continue
                    for ri, row in enumerate(tb.rows):
                        hit = next(
                            (ci for ci, cell in enumerate(row) if cell is not None and _in_bbox(cx, cy, cell)),
                            None,
                        )
                        if hit is not None:
                            membership[li] = (ti, ri, hit)
                            cell_lines.setdefault((ti, ri, hit), []).append(ln)
                            break
                    if li in membership:
                        break

        emitted_tables: set[int] = set()
        skip: set[int] = set()
        for li, line in enumerate(lines):
            if li in skip:
                continue
            if li in membership:
                ti = membership[li][0]
                if ti not in emitted_tables:
                    emitted_tables.add(ti)
                    doc.elements.extend(_table_elements(tables[ti], cell_lines, tracker, page_num, ti))
                continue

            text = clean_text(line.text).strip()
            if not text or _is_header_footer(text, page_num, page_height, line.y0, line.first_color):
                continue

            if tracker.in_future_appendix:
                caption = _future_caption(text)
                if caption:
                    tracker.set_effective_year(*caption)
                    item_stack = []
                    continue

            kind = _heading_kind(line, section_code)
            if kind:
                name, ident, title = kind
                if name == "appendix" and tracker.in_future_appendix:
                    tracker.enter_target_appendix(ident)
                elif name == "appendix":
                    tracker.enter_appendix(ident, title or "")
                elif name == "article":
                    tracker.push(ident, title)
                else:
                    if title is None and li + 1 < len(lines):
                        nl = lines[li + 1]
                        nl_text = clean_text(nl.text).strip()
                        if (
                            abs(nl.y0 - line.y0) < 3
                            and nl.x0 > TITLE_X_MIN
                            and nl_text
                            and _heading_kind(nl, section_code) is None
                            and not nl.first_struck
                            and (ident.count(".") <= 1 or nl.first_bold)
                        ):
                            title = nl_text
                            skip.add(li + 1)
                    tracker.push(ident, title)
                item_stack = []
                continue

            item = _item_level(line)
            item_start = False
            if item:
                level, label, _ = item
                item_stack = item_stack[: level - 1] + [label]
                item_start = True
            doc.elements.extend(_make_elements(line, tracker, ".".join(item_stack), item_start))

    total_pages = pdf.page_count
    pdf.close()
    logger.info(
        "Parsed %s: %d elements across %d pages (%d front-matter pages skipped)",
        document_id,
        len(doc.elements),
        total_pages,
        front_pages,
    )
    return doc
