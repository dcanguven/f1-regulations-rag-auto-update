from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

from app.chunking.chunker import (
    _ArticleGroup,
    _make_chunk_id,
    chunk_document,
    extract_deleted,
)
from app.config import settings
from app.generation.prompt import build_context_block
from app.models.schemas import ParsedDocument, ParsedElement
from app.parsing.pdf_parser import (
    HierarchyTracker,
    _heading_kind,
    _is_header_footer,
    _is_struck,
    _Line,
    _Seg,
    classify_colour,
    parse_pdf,
)
from app.parsing.text_repair import normalize_unicode, repair_corrupted_words
from app.storage.qdrant_store import (
    chunk_to_point,
    ensure_collection,
    scroll_by_article,
    search_dense,
)
from app.storage.sqlite_store import SQLiteStore

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "2026"
PDFS = {"A": (RAW / "A" / "issue_03.pdf", 3), "B": (RAW / "B" / "issue_08.pdf", 8), "C": (RAW / "C" / "issue_20.pdf", 20)}
needs_pdfs = pytest.mark.skipif(not all(p.exists() for p, _ in PDFS.values()), reason="local FIA PDFs missing")


def _line(text, x0=42.5, bold=True, cls="normal", struck=False, color=0x002D5F):
    return _Line(
        page=5, x0=x0, y0=100.0, y1=110.0,
        segs=[_Seg(text, cls, struck, color, 11.0)],
        first_bold=bold, first_color=color, size=11.0,
    )


def _el(text, **kw):
    base = dict(
        text=text, page=1, hierarchy_path="B / B5", article_id="B5",
        article_title="T", content_role="regulatory_text",
    )
    base.update(kw)
    return ParsedElement(**base)


def _doc(elements):
    return ParsedDocument(
        document_id="doc", season=2026, section_code="B", section_name="Sporting",
        issue=8, source_url="u", elements=elements,
    )


@pytest.mark.parametrize("hex_color, expected", [
    (0xFF00FF, "added"), (0xFF40FF, "added"),
    (0xC00000, "governance"), (0xB02418, "governance"),
    (0xFF9900, "reference"), (0xFF9A00, "reference"), (0xFFC000, "reference"),
    (0x00B050, "comment"),
    (0x000000, "normal"), (0x002D5F, "normal"), (0x002060, "normal"),
    (0xFFFFFF, "normal"), (0x467886, "normal"), (0xA6A6A6, "normal"),
])
def test_colour_families_follow_document_legend(hex_color, expected):
    assert classify_colour(hex_color) == expected


def test_legend_swatch_takes_precedence_per_document():
    legend = {"governance": [0xB02418], "added": [0xFF00FF]}
    assert classify_colour(0xB02418, legend) == "governance"
    assert classify_colour(0xFF40FF, legend) == "added"


@pytest.mark.parametrize("text, expected_kind, expected_id", [
    ("C17.3 Standard Supply Components (SSC)", "clause", "C17.3"),
    ("C3.6.2", "clause", "C3.6.2"),
    ("ARTICLE C2: CONVENTIONS AND FUNDAMENTAL DIMENSIONS", "article", "C2"),
    ("APPENDIX C1: DEFINITIONS", "appendix", "C1"),
])
def test_genuine_headings_detected(text, expected_kind, expected_id):
    kind = _heading_kind(_line(text), "C")
    assert kind is not None and kind[0] == expected_kind and kind[1] == expected_id


@pytest.mark.parametrize("line", [
    _line("C3.6.2.c and must conform to the geometries specified below.", x0=92.1, bold=False, color=0),
    _line("C10.7.6, no part of the car may lie more than 950mm", x0=92.1, bold=False, color=0),
    _line("C3.16.1", x0=101.1, bold=False, color=0),
    _line("C1.1", x0=56.8, bold=False, color=0),
    _line("C2.1.1", x0=92.1, bold=True),
    _line("C3.6.2", struck=True, cls="added", color=0xFF00FF),
    _line("A8. Final Notice means", bold=False, color=0),
])
def test_cross_references_and_toc_rows_are_not_headings(line):
    assert _heading_kind(line, "C") is None


def test_section_f_team_text_is_not_a_heading():
    assert _heading_kind(_line("F1 Team personnel", bold=False, color=0), "F") is None


def test_strike_geometry_requires_overlap_at_mid_height():
    rects = [(100.0, 140.0, 105.0)]
    assert _is_struck((110.0, 100.0, 118.0, 110.0), rects) is True
    assert _is_struck((110.0, 120.0, 118.0, 130.0), rects) is False
    assert _is_struck((200.0, 100.0, 208.0, 110.0), rects) is False


def test_footer_filter_keeps_black_body_text_near_bottom():
    assert _is_header_footer("The car must", 1, 842.0, 783.5, 0x000000) is False
    assert _is_header_footer("2026 Formula 1: Sporting", 1, 842.0, 783.5, 0x002D5F) is True


def test_tracker_future_appendix_and_nested_target():
    t = HierarchyTracker("B", 2026)
    t.push("B2.5.2", "Race Session Distance")
    assert t.effective_year == 2026 and t.is_current
    t.enter_appendix("B5", "APPROVED CHANGES TO SECTION B FOR SUBSEQUENT YEARS")
    assert not t.is_current and t.effective_year is None
    t.set_effective_year(2027)
    t.push("B2.5.2", None)
    assert t.effective_year == 2027 and not t.is_current
    assert t.path() == "B / APPENDIX B5 / Changes for 2027 / B2 / B2.5 / B2.5.2"
    t.enter_target_appendix("B1")
    assert "APPENDIX B1" in t.path() and not t.is_current


def test_tracker_inherits_clause_title_for_subclauses():
    t = HierarchyTracker("C", 2026)
    t.push("C3", "AERODYNAMIC COMPONENTS")
    t.push("C3.6", "Plank and Skids")
    t.push("C3.6.2", None)
    assert t.current_article_title == "Plank and Skids"
    assert t.article_heading == "C3 AERODYNAMIC COMPONENTS"


def test_nfkc_normalises_ligatures_but_protects_units():
    assert normalize_unicode("speciﬁed ﬂow") == "specified flow"
    assert normalize_unicode("10000mm² and ¼") == "10000mm² and ¼"
    assert normalize_unicode("a…") == "a..."


def test_ligature_repair_is_exact_and_case_preserving():
    assert repair_corrupted_words("eaect on diaerent parts") == "effect on different parts"
    assert repair_corrupted_words("Diaerent. EAECT") == "Different. EFFECT"
    assert repair_corrupted_words("round-oa and pop-oa valves") == "round-off and pop-off valves"
    assert repair_corrupted_words("OA aaz affect effect") == "OA aaz affect effect"
    once = repair_corrupted_words("suaicient")
    assert repair_corrupted_words(once) == once == "sufficient"


def test_deleted_text_excluded_from_chunks_and_preserved_separately():
    doc = _doc([
        _el("Active text stays.", article_id="B5.1", hierarchy_path="B / B5 / B5.1"),
        _el("Deleted wording", article_id="B5.1", hierarchy_path="B / B5 / B5.1", change_status="deleted", is_deleted=True),
    ])
    chunks = chunk_document(doc)
    assert all("Deleted wording" not in c.text and "Deleted wording" not in c.embedding_text for c in chunks)
    deleted = extract_deleted(doc)
    assert [d.text for d in deleted] == ["Deleted wording"]
    assert deleted[0].article_id == "B5.1" and deleted[0].issue == 8


def test_added_text_is_active_and_marks_chunk_changed():
    doc = _doc([
        _el("Old text ", line_end=False),
        _el("new pink words", change_status="added", line_end=True),
    ])
    (chunk,) = chunk_document(doc)
    assert "new pink words" in chunk.text
    assert chunk.contains_changes and chunk.changed_in_issue == 8 and chunk.is_binding


def test_green_comment_is_separate_non_binding_chunk():
    doc = _doc([
        _el("The car must weigh 800kg."),
        _el("[to be reviewed]", content_role="non_regulatory_comment"),
    ])
    chunks = chunk_document(doc)
    binding = [c for c in chunks if c.is_binding]
    comments = [c for c in chunks if not c.is_binding]
    assert len(binding) == 1 and len(comments) == 1
    assert "to be reviewed" not in binding[0].text
    assert comments[0].content_role == "non_regulatory_comment"
    assert "Non-binding" in comments[0].embedding_text


def test_governance_and_document_references_become_metadata():
    doc = _doc([
        _el("Advisory Committee: SAC", content_role="governance_info"),
        _el("See ", line_end=False),
        _el("FIA-F1-DOC-079", content_role="reference_info", line_end=False),
        _el("."),
    ])
    (chunk,) = chunk_document(doc)
    assert chunk.governance == "Advisory Committee: SAC"
    assert "Advisory Committee" not in chunk.text
    assert chunk.referenced_documents == ["FIA-F1-DOC-079"]


def test_repeated_article_ids_do_not_collide_across_years():
    current = _el("Current rule.", article_id="B2.5.2", hierarchy_path="B / B2 / B2.5 / B2.5.2", effective_year=2026)
    future27 = _el("Rule for 2027.", article_id="B2.5.2", hierarchy_path="B / APPENDIX B5 / Changes for 2027 / B2 / B2.5 / B2.5.2",
                   appendix_id="B5", effective_year=2027, is_current=False)
    future28 = _el("Rule for 2028.", article_id="B2.5.2", hierarchy_path="B / APPENDIX B5 / Changes for 2028 / B2 / B2.5 / B2.5.2",
                   appendix_id="B5", effective_year=2028, is_current=False)
    chunks = chunk_document(_doc([current, future27, future28]))
    assert len(chunks) == 3
    assert len({c.chunk_id for c in chunks}) == 3
    assert [(c.effective_year, c.is_current) for c in chunks] == [(2026, True), (2027, False), (2028, False)]
    assert "effective 2027" in chunks[1].embedding_text


def test_nested_items_start_new_units_and_keep_item_path():
    doc = _doc([
        _el("a.", item_path="a", item_start=True),
        _el("First item text."),
        _el("i.", item_path="a.i", item_start=True),
        _el("Nested item text."),
    ])
    (chunk,) = chunk_document(doc)
    assert chunk.item_path == "a"
    assert "First item text." in chunk.text and "Nested item text." in chunk.text
    assert chunk.text.count("\n") >= 1


def test_chunks_fit_embedding_sequence_limit():
    from app.chunking.chunker import _get_tokenizer

    long_text = " ".join(f"word{i}" for i in range(900))
    chunks = chunk_document(_doc([_el(long_text)]))
    tok = _get_tokenizer()
    assert len(chunks) > 1
    for c in chunks:
        assert len(tok(c.embedding_text)["input_ids"]) <= settings.embedding_max_seq_length


def test_chunk_id_depends_on_scope():
    assert _make_chunk_id("d", "B2.5", 0, "B5|2027|x") != _make_chunk_id("d", "B2.5", 0, "None|2026|y")


def test_group_keeps_split_compatibility():
    g = _ArticleGroup(article_id="B1", article_title=None, hierarchy_path="B / B1")
    g.elements.append(_el("x"))
    assert g.text == "x"


def test_prompt_labels_comments_and_future_changes():
    block = build_context_block([
        {"payload": {"section_code": "B", "article_id": "B2.5", "article_title": "Race", "text": "t",
                     "is_binding": False, "is_current": False, "effective_year": 2027}},
    ])
    assert "NON-BINDING" in block and "2027" in block


def _point_chunk(article_id, year, text="t"):
    from app.models.schemas import Chunk

    return Chunk(
        chunk_id=_make_chunk_id("d", article_id, hash((article_id, year)) % 1000, f"s{year}"),
        document_id="d", season=2026, section_code="B", section_name="Sporting", issue=8,
        article_id=article_id, article_title=None, hierarchy_path="B", page_start=1, page_end=1,
        content_type="article", content_role="regulatory_text", source_url="u", text=text,
        embedding_text=text, article_path=[".".join(article_id.split(".")[: i + 1]) for i in range(len(article_id.split(".")))],
        effective_year=year, is_current=year == 2026,
    )


def test_qdrant_filters_separate_effective_years_and_match_descendants():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    ensure_collection(client, "t", 4)
    specs = [("B2.5.2", 2026), ("B2.5.2", 2027), ("B2.5", 2026), ("B2.50", 2026)]
    chunks = [_point_chunk(a, y, f"{a}-{y}") for a, y in specs]
    client.upsert("t", points=[chunk_to_point(c, [1.0, 0.0, 0.0, 0.0]) for c in chunks])

    hits = scroll_by_article(client, "t", "B2.5", effective_year=2026)
    assert {h["payload"]["article_id"] for h in hits} == {"B2.5", "B2.5.2"}
    assert all(h["payload"]["effective_year"] == 2026 for h in hits)
    assert len(scroll_by_article(client, "t", "B2.5", effective_year=2027)) == 1
    exact = scroll_by_article(client, "t", "B2.5", effective_year=2026, descendants=False)
    assert {h["payload"]["article_id"] for h in exact} == {"B2.5"}
    dense = search_dense(client, "t", [1.0, 0.0, 0.0, 0.0], 10, effective_year_filter=2027)
    assert [d["payload"]["effective_year"] for d in dense] == [2027]


def test_sqlite_deleted_provisions_roundtrip(tmp_db):
    doc = _doc([_el("Gone", change_status="deleted", is_deleted=True)])
    with SQLiteStore(tmp_db) as db:
        db.replace_deleted_provisions("doc", extract_deleted(doc))
        db.replace_deleted_provisions("doc", extract_deleted(doc))
        rows = db.list_deleted_provisions("doc")
    assert len(rows) == 1 and rows[0]["text"] == "Gone"


@pytest.fixture(scope="module")
def parsed():
    out = {}
    for sec, (path, issue) in PDFS.items():
        doc = parse_pdf(path, f"fia_f1_2026_{sec}_issue_{issue:02d}", 2026, sec, sec, issue, "u")
        out[sec] = (doc, chunk_document(doc), extract_deleted(doc))
    return out


@needs_pdfs
def test_pdf_legends_are_read_per_document(parsed):
    assert parsed["A"][0].legend["governance"] == ["#c00000"]
    assert parsed["B"][0].legend["governance"] == ["#b02418"]
    assert all(parsed[s][0].legend["added"] == ["#ff00ff"] for s in "ABC")


@needs_pdfs
def test_pdf_front_matter_and_furniture_excluded(parsed):
    assert [parsed[s][0].front_matter_pages for s in "ABC"] == [3, 3, 6]
    for sec in "ABC":
        _, chunks, _ = parsed[sec]
        assert min(c.page_start for c in chunks) >= parsed[sec][0].front_matter_pages + 1
        assert not any(re.search(r"©\s?2026|SECTION [A-F]: [A-Z]|ARTICLE [A-F]\d+:", c.text) for c in chunks)


@needs_pdfs
def test_pdf_cross_references_do_not_create_articles(parsed):
    _, chunks, _ = parsed["C"]
    ids = {c.article_id for c in chunks}
    assert "C3.6.2.c" not in ids
    assert any("C3.6.2.c and must conform" in c.text for c in chunks if c.article_id == "C3.6.3")
    depth1 = collections.Counter(c.article_id for c in chunks if c.article_id and "." not in c.article_id)
    assert not depth1


@needs_pdfs
def test_pdf_page_references_match_printed_pages(parsed):
    def first_page(sec, aid):
        return min(c.page_start for c in parsed[sec][1] if c.article_id == aid and c.is_current)
    assert first_page("A", "A1.1.1") == 5
    assert first_page("B", "B1.6.3") == 10
    assert first_page("C", "C3.6.2") == 24


@needs_pdfs
def test_pdf_mixed_colour_line_keeps_added_segment(parsed):
    elements = [e for e in parsed["A"][0].elements if e.page == 76 and ", if applicable," in e.text]
    assert elements and elements[0].change_status == "added" and elements[0].color_hex == "#ff40ff"
    same_line = [e for e in parsed["A"][0].elements if e.page == 76 and e.change_status == "unchanged"]
    assert same_line


@needs_pdfs
def test_pdf_added_provisions_flagged_changed(parsed):
    changed = [c for c in parsed["B"][1] if c.contains_changes]
    assert changed and all(c.changed_in_issue == 8 for c in changed)
    assert any("Recharge limit" in c.text and c.article_id for c in changed) or any(c.page_start == 64 for c in changed)


@needs_pdfs
def test_pdf_deleted_provisions_preserved_and_excluded(parsed):
    deleted_c = [d for d in parsed["C"][2] if d.page == 164]
    assert any("low-level impurities" in d.text for d in deleted_c)
    active_c = [c for c in parsed["C"][1] if c.page_start <= 164 <= c.page_end]
    assert not any("to cover the presence of low-level impurities" in c.text for c in active_c)
    assert any(d.page == 84 and d.appendix_id == "A9" and "Power unit maximum supply price" in d.text for d in parsed["A"][2])
    assert any(d.page == 78 and "modifications already approved by the FIA" in d.text for d in parsed["B"][2])


@needs_pdfs
def test_pdf_green_comments_are_non_binding(parsed):
    a = [c for c in parsed["A"][1] if not c.is_binding]
    assert any("driver definitions to be reviewed" in c.text for c in a)
    c_comments = [c for c in parsed["C"][1] if not c.is_binding]
    assert any("FIA-F1-DOC-042" in c.text for c in c_comments)
    for sec in "ABC":
        assert not any("driver definitions to be reviewed" in c.text for c in parsed[sec][1] if c.is_binding)


@needs_pdfs
def test_pdf_future_appendices_carry_effective_year(parsed):
    b = parsed["B"][1]
    same_id = [c for c in b if c.article_id == "B2.5.2"]
    assert {(c.effective_year, c.is_current) for c in same_id} >= {(2026, True), (2027, False), (2028, False)}
    assert len({c.chunk_id for c in same_id}) == len(same_id)
    future_a = {c.effective_year for c in parsed["A"][1] if not c.is_current}
    assert future_a == {2028, 2029, 2030}
    assert all(c.appendix_id == "B5" for c in b if not c.is_current)
    assert all(c.is_current for c in parsed["C"][1])


@needs_pdfs
def test_pdf_tables_keep_header_value_relationships(parsed):
    rows = [c for c in parsed["C"][1] if c.contains_table and c.page_start <= 49 <= c.page_end]
    assert rows
    text = " ".join(c.text for c in rows)
    assert "C3.16.1 Nose" in text and "(a.i) / (a.iii): [15, 40, 20] [0°, ±10°, 0°]" in text and "(d): 750" in text
    assert any(c.content_type == "table" for c in rows)


@needs_pdfs
def test_pdf_ligatures_and_corrupted_words_fixed(parsed):
    for sec in "ABC":
        text = " ".join(c.text for c in parsed[sec][1])
        assert not re.search("[ﬁﬂ]", text)
        assert not re.search(r"\b(?:eaect\w*|diaer\w*|aaect\w*|suaicient\w*|stianess)\b", text, re.I)
    c_text = " ".join(c.text for c in parsed["C"][1])
    assert "different" in c_text and "effect" in c_text
    assert c_text.count("²") == 12


@needs_pdfs
def test_pdf_chunks_fit_sequence_limit(parsed):
    from app.chunking.chunker import _get_tokenizer

    tok = _get_tokenizer()
    for sec in "ABC":
        lengths = [len(x) for x in tok([c.embedding_text for c in parsed[sec][1]])["input_ids"]]
        assert max(lengths) <= settings.embedding_max_seq_length


LATEST = {
    "A": (RAW / "A" / "issue_03.pdf", 3),
    "B": (RAW / "B" / "issue_09.pdf", 9),
    "C": (RAW / "C" / "issue_21.pdf", 21),
    "D": (RAW / "D" / "issue_08.pdf", 8),
    "E": (RAW / "E" / "issue_07.pdf", 7),
    "F": (RAW / "F" / "issue_11.pdf", 11),
}
needs_latest = pytest.mark.skipif(not all(p.exists() for p, _ in LATEST.values()), reason="latest FIA PDFs missing")


@pytest.fixture(scope="module")
def latest():
    from app.ingestion.catalog import SECTION_NAMES

    out = {}
    for sec, (path, issue) in LATEST.items():
        doc = parse_pdf(path, f"fia_f1_2026_{sec}_issue_{issue:02d}", 2026, sec, SECTION_NAMES[sec], issue, "u")
        out[sec] = (doc, chunk_document(doc), extract_deleted(doc))
    return out


def test_future_caption_formats():
    from app.parsing.pdf_parser import _future_caption

    assert _future_caption("Changes for 2027") == (2027, None)
    assert _future_caption(
        "Approved Changes for Financial Regulations Reporting Periods starting from 1 January 2027"
    ) == (2027, "1 January 2027")
    assert _future_caption(
        "Approved changes for Financial Regulations Reporting Periods starting from 1 January 2028 "
    ) == (2028, "1 January 2028")
    assert _future_caption("Starting from 1 January 2027, Entities Subject To Segregation Requirements shall:") is None
    assert _future_caption("The Changes for the 2027 season are listed") is None


def test_header_row_guard_rejects_sentences_only():
    from app.parsing.pdf_parser import _looks_like_header

    assert _looks_like_header(["Name", "Code (RV-)", "Point 1 [X, Y, Z]"])
    assert _looks_like_header(["33.", "Name", ""])
    assert not _looks_like_header(["2.1 In accordance with Article C3.10.10, adjustment of the position of the Front Wing Profiles."])


@needs_latest
def test_front_matter_pages_for_all_sections(latest):
    assert {s: latest[s][0].front_matter_pages for s in "ABCDEF"} == {"A": 3, "B": 3, "C": 6, "D": 3, "E": 2, "F": 2}


@needs_latest
def test_effective_year_classification_in_all_sections(latest):
    for sec in "ABCDEF":
        for c in latest[sec][1]:
            if c.is_current:
                assert c.effective_year == 2026
            else:
                assert c.effective_year is not None and c.effective_year >= 2027, (sec, c.page_start)
    years = {s: {c.effective_year for c in latest[s][1] if not c.is_current} for s in "ABCDEF"}
    assert years == {"A": {2028, 2029, 2030}, "B": {2027, 2028, 2029}, "C": set(), "D": {2027, 2028}, "E": {2028}, "F": {2027, 2028, 2029}}
    assert {c.effective_from for c in latest["D"][1] if not c.is_current} == {"1 January 2027", "1 January 2028"}
    assert {c.effective_from for c in latest["E"][1] if not c.is_current} == {"1 January 2028"}


@needs_latest
def test_section_f_current_and_future_split(latest):
    chunks = latest["F"][1]
    future = [c for c in chunks if not c.is_current]
    current = [c for c in chunks if c.is_current]
    assert len(future) == 63 and len(current) == 121
    assert all(c.appendix_id == "F2" for c in future)
    assert min(c.page_start for c in chunks) >= 3


@needs_latest
def test_section_f_wind_tunnel_table_keeps_caption_and_value(latest):
    text = " ".join(c.text for c in latest["F"][1] if c.contains_table and c.page_start <= 16 <= c.page_end)
    assert "Wind tunnel limits for C=100%: RWTT Runs | # | 320" in text
    assert "CFD limits for C=100%: 3D new RATGs used for solve or solve part of all RCFDs | # | 2000" in text


@needs_latest
def test_financial_currency_tables_map_values_to_currencies(latest):
    d = " ".join(c.text for c in latest["D"][1] if c.contains_table and c.page_start <= 8 <= c.page_end)
    assert "US Dollars (‘000): 215’000" in d and "Pounds Sterling (‘000): 170’090" in d
    assert "Euros (‘000): 198’663" in d and "Swiss Francs (‘000): 189’992" in d
    e = " ".join(c.text for c in latest["E"][1] if c.contains_table and c.page_start <= 6 <= c.page_end)
    assert "US Dollars (‘000): 190,000" in e and "Japanese Yen (‘000): 24,967,900" in e


@needs_latest
def test_text_rows_are_not_used_as_column_labels(latest):
    rows = " ".join(c.text for c in latest["B"][1] if c.contains_table and 87 <= c.page_start <= 89)
    assert "1. BRAKES 1.1 Brake friction material may be removed" in rows
    assert not re.search(r"replaced\.: Any", rows)


@needs_latest
def test_table_numbers_exist_on_source_pages(latest):
    import pymupdf

    for sec, (path, _) in LATEST.items():
        pdf = pymupdf.open(path)
        raw = {n: re.sub(r"\s+", " ", pdf[n - 1].get_text()) for n in range(1, pdf.page_count + 1)}
        for c in latest[sec][1]:
            if not c.contains_table:
                continue
            pages = " ".join(raw[n] for n in range(c.page_start, c.page_end + 1))
            for tok in set(re.findall(r"\d[\d’',.\-−]*\d|\d", c.text)):
                assert tok in pages, (sec, c.page_start, tok)


@needs_latest
def test_current_year_retrieval_excludes_future_provisions(latest):
    from qdrant_client import QdrantClient

    from app.retrieval.sparse import BM25Index

    client = QdrantClient(":memory:")
    ensure_collection(client, "t", 4)
    chunks = [c for s in "BDEF" for c in latest[s][1]]
    client.upsert("t", points=[chunk_to_point(c, [1.0, 0.0, 0.0, 0.0]) for c in chunks])
    assert any(not c.is_current for c in chunks)

    dense = search_dense(client, "t", [1.0, 0.0, 0.0, 0.0], 5000, effective_year_filter=2026)
    assert dense and all(d["payload"]["is_current"] and d["payload"]["effective_year"] == 2026 for d in dense)
    future = search_dense(client, "t", [1.0, 0.0, 0.0, 0.0], 5000, effective_year_filter=2027)
    assert future and all(not d["payload"]["is_current"] and d["payload"]["effective_year"] == 2027 for d in future)

    exact = scroll_by_article(client, "t", "B2.5", section_code="B", effective_year=2026)
    assert exact and all(h["payload"]["is_current"] for h in exact)

    bm25 = BM25Index()
    bm25.build([{**vars_(c)} for c in chunks], [c.chunk_id for c in chunks])
    hits = bm25.search("race session distance wind tunnel reporting period", 200, effective_year_filter=2026)
    assert hits and all(h["payload"]["is_current"] for h in hits)


def vars_(chunk):
    from dataclasses import asdict

    return asdict(chunk)
