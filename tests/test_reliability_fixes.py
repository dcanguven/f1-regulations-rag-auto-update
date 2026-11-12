from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf
import pytest

from app.chunking.chunker import chunk_document, extract_deleted
from app.config import settings
from app.ingestion.catalog import SECTION_NAMES
from app.models.schemas import Chunk, QueryScope
from app.parsing.pdf_parser import _is_header_footer, parse_pdf
from app.retrieval.query_processor import extract_scope, normalize_article
from app.retrieval.retriever import maybe_expand, select_exact_context, validate_scope
from app.storage.qdrant_store import (
    chunk_to_point,
    document_has_points,
    document_index_current,
    ensure_collection,
    known_article_ids,
    prune_obsolete_chunks,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "2026"
PDFS = {
    "A": (RAW / "A" / "issue_03.pdf", 3),
    "B": (RAW / "B" / "issue_09.pdf", 9),
    "D": (RAW / "D" / "issue_08.pdf", 8),
    "E": (RAW / "E" / "issue_07.pdf", 7),
    "F": (RAW / "F" / "issue_11.pdf", 11),
}
needs_pdfs = pytest.mark.skipif(not all(p.exists() for p, _ in PDFS.values()), reason="latest FIA PDFs missing")


@pytest.mark.parametrize("question, expected", [
    ("Article B5.13", "B5.13"),
    ("B5.13", "B5.13"),
    ("What does Article B5.13 say?", "B5.13"),
    ("Madde B 5 13", "B5.13"),
    ("Madde B 5 13 nedir", "B5.13"),
    ("What does B 1 6 3 say?", "B1.6.3"),
    ("B 5 13 maddesini açıklar mısınız?", "B5.13"),
    ("Show me article A2.2.1", "A2.2.1"),
    ("What does article C14 say?", "C14"),
    ("section B article 5.13.4", "B5.13.4"),
    ("Article C14.3.a", "C14.3.a"),
])
def test_valid_article_references_are_normalised_without_adjacent_words(question, expected):
    assert normalize_article(question) == expected
    scope = extract_scope(question)
    assert scope.exact_article == expected


@pytest.mark.parametrize("question", [
    "What is F1?",
    "What is the cost cap for F1 teams in US dollars for 2026?",
    "What is the minimum mass of an F1 car?",
    "How many seconds are added for a 5-Second Penalty served after a TTCS?",
    "Is there a 5 second penalty for speeding in the pit lane?",
    "What coefficient C applies to a team that finished 5th in the constructors championship?",
    "Who won the 2023 Formula 1 World Championship?",
    "Bir Formula 1 aracının minimum kütlesi nedir?",
    "Takımlar için 2026 maliyet sınırı ABD doları cinsinden nedir?",
    "a 5 10 something",
    "What is C4?",
])
def test_ordinary_language_is_not_an_article_reference(question):
    scope = extract_scope(question)
    assert scope.exact_article is None
    assert scope.section_code is None
    assert scope.section_from_article is False


def test_explicit_section_is_kept_without_an_article():
    scope = extract_scope("According to Section B, how many cars can a team enter?")
    assert scope.section_code == "B" and scope.exact_article is None


def _chunk(article_id, version="2", doc="d1", index=0, text="t"):
    return Chunk(
        chunk_id=hashlib.md5(f"{doc}|{article_id}|{index}|{version}".encode()).hexdigest(),
        document_id=doc, season=2026, section_code="B", section_name="Sporting", issue=9,
        article_id=article_id, article_title=None, hierarchy_path="B", page_start=1, page_end=1,
        content_type="article", content_role="regulatory_text", source_url="u", text=text,
        embedding_text=text,
        article_path=[".".join(article_id.split(".")[: i + 1]) for i in range(len(article_id.split(".")))],
        effective_year=2026, pipeline_version=version,
    )


def _memory_index(chunks):
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    ensure_collection(client, "t", 4)
    client.upsert("t", points=[chunk_to_point(c, [1.0, 0.0, 0.0, 0.0]) for c in chunks])
    return client


def test_unknown_article_is_cleared_together_with_its_inferred_section():
    client = _memory_index([_chunk("B5.13"), _chunk("B5.13.1")])
    settings_collection = settings.qdrant_collection
    settings.qdrant_collection = "t"
    try:
        known = known_article_ids(client, "t")
        assert {"B5", "B5.13", "B5.13.1"} <= known
        scope = extract_scope("What does Article B99.99 say?")
        assert scope.exact_article == "B99.99" and scope.section_code == "B"
        validate_scope(scope, client)
        assert scope.exact_article is None and scope.section_code is None
        scope = extract_scope("What does Article B5.13 say?")
        validate_scope(scope, client)
        assert scope.exact_article == "B5.13" and scope.section_code == "B"
        explicit = extract_scope("In Section C what does Article B99.99 say?")
        validate_scope(explicit, client)
        assert explicit.exact_article is None and explicit.section_code == "C"
    finally:
        settings.qdrant_collection = settings_collection


def _hit(article_id, page, tokens=100, vec=None):
    return {"id": f"{article_id}-{page}", "score": 1.0, "vector": vec,
            "payload": {"article_id": article_id, "page_start": page, "text": "x" * (tokens * 4)}}


def test_exact_context_budget_limits_chunks_and_keeps_requested_article_first():
    hits = [_hit("B5.1", 1)] + [_hit(f"B5.{i}.1", i) for i in range(2, 40)] + [_hit("B5", 50)]
    chosen, truncated = select_exact_context(hits, "B5", None, 12, 100000)
    assert truncated and len(chosen) == 12
    assert any(h["payload"]["article_id"] == "B5" for h in chosen)
    pages = [h["payload"]["page_start"] for h in chosen]
    assert pages == sorted(pages)


def test_exact_context_budget_respects_token_limit_and_similarity():
    near = [1.0, 0.0]
    far = [0.0, 1.0]
    hits = [_hit(f"C3.{i}", i, tokens=500, vec=far) for i in range(1, 6)] + [_hit("C3.9", 9, tokens=500, vec=near)]
    chosen, truncated = select_exact_context(hits, "C3", near, 12, 1100)
    assert truncated
    assert sum(len(h["payload"]["text"]) // 4 for h in chosen) <= 1100
    assert any(h["payload"]["article_id"] == "C3.9" for h in chosen)


def test_specific_clause_lookup_is_not_truncated():
    hits = [_hit("B5.13.8", 47, tokens=150), _hit("B5.13.8", 48, tokens=150)]
    chosen, truncated = select_exact_context(hits, "B5.13.8", None, 12, 2400)
    assert not truncated and chosen == hits


def test_single_oversized_chunk_is_still_returned():
    chosen, truncated = select_exact_context([_hit("C1.1", 1, tokens=9000), _hit("C1.2", 2, tokens=10)], "C1", None, 12, 100)
    assert truncated and len(chosen) >= 1


def test_stale_pipeline_versions_are_detected_and_obsolete_chunks_pruned():
    old = [_chunk("B1.1", version="", index=0), _chunk("B1.2", version="", index=1)]
    other = [_chunk("A1.1", version="2", doc="other")]
    client = _memory_index(old + other)
    assert document_has_points(client, "t", "d1")
    assert not document_index_current(client, "t", "d1", "2")
    assert document_index_current(client, "t", "other", "2")
    fresh = [_chunk("B1.1", version="2", index=5), _chunk("B1.3", version="2", index=6)]
    client.upsert("t", points=[chunk_to_point(c, [1.0, 0.0, 0.0, 0.0]) for c in fresh])
    assert not document_index_current(client, "t", "d1", "2")
    removed = prune_obsolete_chunks(client, "t", "d1", {c.chunk_id for c in fresh})
    assert removed == 2
    assert document_index_current(client, "t", "d1", "2")
    assert document_has_points(client, "t", "other")
    remaining = {str(p.id).replace("-", "") for p in client.scroll("t", limit=50)[0]}
    assert {c.chunk_id for c in fresh} <= remaining and not ({c.chunk_id for c in old} & remaining)


def test_footer_lines_are_recognised_in_the_section_f_band():
    height = 841.92
    assert _is_header_footer("01 October 2026", 17, height, 773.2, 0x002D5F)
    assert _is_header_footer("2026 Formula 1 Regulations - Section F [Operational]", 17, height, 773.0, 0x002D5F)
    assert _is_header_footer("©2026 Fédération Internationale de l’Automobile", 17, height, 784.0, 0x002D5F)
    assert _is_header_footer("F17", 17, height, 776.2, 0xFFFFFF)
    assert not _is_header_footer("The car must weigh at least 724kg.", 17, height, 776.0, 0x000000)
    assert not _is_header_footer("C3.6.2", 17, height, 776.0, 0x002060)
    assert not _is_header_footer("F17", 17, height, 400.0, 0xFFFFFF)


@pytest.fixture(scope="module")
def parsed():
    out = {}
    for sec, (path, issue) in PDFS.items():
        doc = parse_pdf(path, f"fia_f1_2026_{sec}_issue_{issue:02d}", 2026, sec, SECTION_NAMES[sec], issue, "u")
        out[sec] = (doc, chunk_document(doc))
    return out


def _table_rows(parsed, sec, page):
    return [e.text for e in parsed[sec][0].elements if e.is_table and e.page == page]


@needs_pdfs
def test_section_f_footer_text_is_gone_and_footer_only_chunks_do_not_exist(parsed):
    doc, chunks = parsed["F"]
    footer = re.compile(r"Formula 1 Regulations - Section F|\d{1,2} (?:June|August|October) 2026|©\s?2026|\bF\d{1,2}\b 01 October")
    assert not [c for c in chunks if footer.search(c.text)]
    assert not [e for e in doc.elements if re.fullmatch(r"F\d{1,2}", e.text.strip())]
    assert not [c for c in chunks if re.fullmatch(r"(?:F\d{1,2}\s*)?(?:\d{1,2} \w+ 20\d\d\s*)?", c.text.strip())]


@needs_pdfs
def test_section_f_content_ids_and_pages_survive_footer_removal(parsed):
    chunks = parsed["F"][1]
    def pages(aid):
        return {(c.page_start, c.page_end) for c in chunks if c.article_id == aid and c.is_current}
    assert any(p[0] == 6 for p in pages("F3.2.1"))
    assert any(p[0] == 17 for p in pages("F4.6.4"))
    assert any(p[0] == 16 for p in pages("F4.6.1"))
    assert any("fourteen (14) consecutive calendar days" in c.text for c in chunks if c.article_id == "F3.2.1")
    assert len([c for c in chunks if c.is_current]) >= 120


@needs_pdfs
def test_other_sections_have_no_footer_regressions(parsed):
    for sec in "ABDE":
        assert not [c for c in parsed[sec][1] if re.search(r"Formula 1 Regulations - Section [A-E]|Issue \d\d$", c.text)]


A_P10_FIRST = [
    ("1st", [("≥ 2L", "6"), ("≥25%", "13"), ("≥50%", "19"), ("≥75%", "25")]),
    ("2nd", [("≥ 2L", "4"), ("≥25%", "10"), ("≥50%", "14"), ("≥75%", "18")]),
    ("3rd", [("≥ 2L", "3"), ("≥25%", "8"), ("≥50%", "12"), ("≥75%", "15")]),
    ("4th", [("≥ 2L", "2"), ("≥25%", "6"), ("≥50%", "10"), ("≥75%", "12")]),
    ("5th", [("≥ 2L", "1"), ("≥25%", "5"), ("≥50%", "8"), ("≥75%", "10")]),
    ("6th", [("≥25%", "4"), ("≥50%", "6"), ("≥75%", "8")]),
    ("7th", [("≥25%", "3"), ("≥50%", "4"), ("≥75%", "6")]),
    ("8th", [("≥25%", "2"), ("≥50%", "3"), ("≥75%", "4")]),
    ("9th", [("≥25%", "1"), ("≥50%", "2"), ("≥75%", "2")]),
    ("10th", [("≥50%", "1"), ("≥75%", "1")]),
]


@needs_pdfs
def test_championship_points_table_rows_match_the_pdf_exactly(parsed):
    rows = _table_rows(parsed, "A", 10)
    race = rows[:10]
    assert len(rows) == 17
    for row, (position, cells) in zip(race, A_P10_FIRST):
        parts = row.split(" | ")
        assert parts[0] == f"Position: {position}"
        values = parts[1:]
        assert len(values) == len(cells), row
        for part, (threshold, points) in zip(values, cells):
            label, value = part.rsplit(": ", 1)
            assert label.endswith(threshold) and value == points, (row, threshold, points)
    sprint = [("1st", "8"), ("2nd", "7"), ("3rd", "6"), ("4th", "5"), ("5th", "4"), ("6th", "3"), ("7th", "2")]
    for row, (position, points) in zip(rows[10:], sprint):
        assert row == f"Position: {position} | Points Col. 1 ≥ 50%: {points}"


@needs_pdfs
def test_repeated_equal_values_in_adjacent_columns_are_preserved(parsed):
    rows = _table_rows(parsed, "A", 10)
    assert rows[8].endswith("Points Col. 3 ≥50%: 2 | Points Col. 4 ≥75%: 2")
    assert rows[9].endswith("Points Col. 3 ≥50%: 1 | Points Col. 4 ≥75%: 1")


@needs_pdfs
def test_tyre_allocation_table_keeps_all_values_and_format_headers(parsed):
    rows = _table_rows(parsed, "B", 58)
    assert len(rows) == 3
    expected = [
        ("Dry-Weather Tyres (excluding evaluation or test specifications)", ["13", "12", "12"]),
        ("Intermediate Tyres", ["5", "5*", "5"]),
        ("Wet-Weather Tyres", ["2", "2", "2"]),
    ]
    formats = ["Standard Format Competition", "Alternative Format Competition", "Standard Format Competition with ICTT"]
    for row, (label, values) in zip(rows, expected):
        parts = row.split(" | ")
        assert parts[0].startswith("Format of the Competition:")
        assert parts[1] == label
        assert parts[2:] == [f"{fmt}: {v}" for fmt, v in zip(formats, values)]
    assert not any("Maximum number of sets" in r and "13 | 12" in r and "Competition" not in r for r in rows)


def _extract_values(pdf_path, page, table_index):
    tf = pymupdf.open(pdf_path)[page - 1].find_tables()
    return tf.tables[table_index].extract()


@needs_pdfs
def test_serialised_rows_contain_every_source_cell_in_order(parsed):
    checks = [("A", 10, 0, 3), ("A", 10, 1, 3), ("B", 58, 1, 1), ("D", 8, 0, 4), ("E", 6, 0, 1)]
    for sec, page, ti, first_data in checks:
        path, _ = PDFS[sec]
        source = _extract_values(path, page, ti)
        rows = _table_rows(parsed, sec, page)
        if sec == "A" and ti == 1:
            rows = rows[10:]
        elif sec == "A":
            rows = rows[:10]
        data = [[c.strip().replace("\n", " ") for c in r if c and c.strip()] for r in source[first_data:]]
        data = [r for r in data if r]
        assert len(data) == len(rows), (sec, page, len(data), len(rows))
        for cells, row in zip(data, rows):
            values = [p.rsplit(": ", 1)[-1] if ": " in p else p for p in row.split(" | ")]
            cursor = 0
            for cell in cells:
                while cursor < len(values) and values[cursor].strip() != cell:
                    cursor += 1
                assert cursor < len(values), (sec, page, cells, row)
                cursor += 1


@needs_pdfs
def test_financial_and_aero_tables_unchanged_by_table_fixes(parsed):
    assert "Presentation Currency | US Dollars (‘000): 215’000 | Pounds Sterling (‘000): 170’090 | Euros (‘000): 198’663 | Swiss Francs (‘000): 189’992" in _table_rows(parsed, "D", 8)
    assert "US Dollars (‘000): 190,000 | Pounds Sterling (‘000): 152,918 | Euros (‘000): 180,420 | Japanese Yen (‘000): 24,967,900" in _table_rows(parsed, "E", 6)
    f16 = _table_rows(parsed, "F", 16)
    assert "Wind tunnel limits for C=100%: RWTT Runs | # | 320" in f16
    assert "CFD limits for C=100%: 3D new RATGs used for solve or solve part of all RCFDs | # | 2000" in f16


@needs_pdfs
def test_pipeline_version_is_stamped_on_every_chunk(parsed):
    for sec in parsed:
        assert {c.pipeline_version for c in parsed[sec][1]} == {settings.pipeline_version}


@needs_pdfs
def test_struck_table_rows_are_kept_as_deleted_provisions(parsed):
    deleted = extract_deleted(parsed["B"][0])
    assert any(d.page == 87 and "not controlled by the FIA Standard ECU" in d.text for d in deleted)
    assert len(deleted) == 46
    p87_active = " ".join(c.text for c in parsed["B"][1] if c.page_start == 87)
    assert ", not controlled by the FIA Standard ECU," not in p87_active


def test_context_expansion_is_off_by_default_and_skips_the_scroll(monkeypatch):
    from app.retrieval import retriever as retriever_module

    calls = []
    monkeypatch.setattr(retriever_module, "expand_context", lambda *a, **k: calls.append(1) or ["expanded"])
    candidates = [{"id": "1", "payload": {}}]
    assert settings.enable_context_expansion is False
    assert maybe_expand(candidates, object()) is candidates and not calls
    monkeypatch.setattr(settings, "enable_context_expansion", True)
    assert maybe_expand(candidates, object()) == ["expanded"] and calls


def _table_units_doc(rows):
    from app.models.schemas import ParsedDocument, ParsedElement

    elements = [
        ParsedElement(text="Intro text before the table.", page=1, hierarchy_path="B / B1", article_id="B1",
                      article_title="T", content_role="regulatory_text")
    ]
    for r in rows:
        elements.append(ParsedElement(text=r, page=1, hierarchy_path="B / B1", article_id="B1", article_title="T",
                                      content_role="regulatory_text", is_table=True, item_start=True))
    return ParsedDocument(document_id="d", season=2026, section_code="B", section_name="Sporting", issue=9,
                          source_url="u", elements=elements)


def test_table_chunk_mode_default_packs_rows_and_row_modes_split_them(monkeypatch):
    rows = [f"Position: {i} | Points: {i}" for i in range(1, 6)]
    assert settings.table_chunk_mode == "packed"
    packed = [c for c in chunk_document(_table_units_doc(rows)) if c.content_type == "table"]
    assert len(packed) == 1
    monkeypatch.setattr(settings, "table_chunk_mode", "row")
    per_row = [c for c in chunk_document(_table_units_doc(rows)) if c.content_type == "table"]
    assert len(per_row) == 5 and all(c.text.startswith("Position: ") for c in per_row)
    monkeypatch.setattr(settings, "table_chunk_mode", "row_small")
    monkeypatch.setattr(settings, "table_row_chunk_max_rows", 3)
    big = [c for c in chunk_document(_table_units_doc(rows)) if c.content_type == "table"]
    assert len(big) == 1
