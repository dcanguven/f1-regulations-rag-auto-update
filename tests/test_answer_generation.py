from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from app.chunking.chunker import chunk_document
from app.config import settings
from app.generation.answer import generate
from app.generation.llm import INSUFFICIENT_EVIDENCE
from app.generation.prompt import build_context_block, build_user_message
from app.generation.verification import verify_claim
from app.ingestion.catalog import SECTION_NAMES
from app.models.schemas import QueryScope
from app.parsing.pdf_parser import parse_pdf
from app.retrieval.query_processor import classify_year, extract_scope
from app.retrieval.retriever import YearResolution, decide_year_scope

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "2026"
PDFS = {"B": (RAW / "B" / "issue_09.pdf", 9), "C": (RAW / "C" / "issue_21.pdf", 21)}
needs_pdfs = pytest.mark.skipif(not all(p.exists() for p, _ in PDFS.values()), reason="latest FIA PDFs missing")
GEN = "app.generation.answer.generate_structured"


@pytest.mark.parametrize("question, year, mode", [
    ("How many times can a team apply for a new ECU in 2028?", 2028, "year_mention"),
    ("How many ECU applications are allowed in 2027?", 2027, "year_mention"),
    ("What is the race distance in 2027?", 2027, "year_mention"),
    ("From 2027, what will count as one run in the wind tunnel?", 2027, "future_explicit"),
    ("How many exceptions will a competitor get in the first restricted period from 2028?", 2028, "future_explicit"),
    ("What changes from 1 January 2028 for power unit inventories?", 2028, "future_explicit"),
    ("Starting in 2029, how are sprint points awarded?", 2029, "future_explicit"),
    ("Will the pit lane rules change in 2027?", 2027, "future_explicit"),
    ("Which approved changes for 2028 affect testing?", 2028, "future_explicit"),
    ("What is the minimum mass in the 2026 season?", 2026, "current"),
    ("Who won the 2023 championship?", 2023, "past"),
    ("What is the minimum mass of the car?", None, "current"),
    ("What is the limit of 2000 runs?", None, "current"),
    ("Is the 2030 allowance different?", 2030, "year_mention"),
    ("How many gearbox changes does a driver get in 2029?", 2029, "year_mention"),
])
def test_year_mode_classification(question, year, mode):
    assert classify_year(question, 2026) == (year, mode)
    scope = extract_scope(question)
    assert scope.year_mode == mode and scope.year_mentioned == year


@pytest.fixture(scope="module")
def chunks():
    out = {}
    for sec, (path, issue) in PDFS.items():
        doc = parse_pdf(path, f"fia_f1_2026_{sec}_issue_{issue:02d}", 2026, sec, SECTION_NAMES[sec], issue, "u")
        out[sec] = chunk_document(doc)
    return out


def _cand(chunk):
    return {"id": chunk.chunk_id, "score": 1.0, "payload": asdict(chunk)}


@needs_pdfs
def test_current_table_that_lists_the_year_is_resolved_as_current(chunks):
    ecu = next(c for c in chunks["C"] if c.article_id == "C8.1.8" and c.is_current and "2028" in c.text)
    future = [c for c in chunks["B"] if not c.is_current and c.effective_year == 2028]
    res = decide_year_scope(2028, [_cand(ecu)], [_cand(c) for c in future])
    assert res.mode == "current" and "2028" in res.reason


@needs_pdfs
def test_year_absent_from_current_text_resolves_to_future_provision(chunks):
    current = next(c for c in chunks["B"] if c.article_id == "B2.5.2" and c.is_current)
    future = next(c for c in chunks["B"] if c.article_id == "B2.5.2" and c.effective_year == 2027)
    assert "2027" not in current.text and "290km" in future.text
    res = decide_year_scope(2027, [_cand(current)], [_cand(future)])
    assert res.mode == "future" and res.explicit is False


@needs_pdfs
def test_same_article_in_current_text_and_future_change_is_ambiguous(chunks):
    current = next(c for c in chunks["B"] if c.article_id == "B2.5.2" and c.is_current)
    future = next(c for c in chunks["B"] if c.article_id == "B2.5.2" and c.effective_year == 2027)
    mentioning = asdict(current)
    mentioning["text"] += " Changes apply in 2027."
    res = decide_year_scope(2027, [{"id": "x", "payload": mentioning}], [_cand(future)])
    assert res.mode == "ambiguous_both" and res.shared_future


def test_no_evidence_for_the_year_stays_unresolved():
    res = decide_year_scope(2031, [{"id": "a", "payload": {"text": "nothing", "article_id": "A1.1"}}], [])
    assert res.mode == "unresolved"


def _memory_retriever(monkeypatch, selected):
    from qdrant_client import QdrantClient

    from app.retrieval import retriever as rm
    from app.retrieval.embedder import Embedder
    from app.retrieval.sparse import BM25Index
    from app.storage.qdrant_store import chunk_to_point, ensure_collection

    vectors = Embedder().encode([c.embedding_text for c in selected])
    client = QdrantClient(":memory:")
    ensure_collection(client, "yr", 384)
    points = [chunk_to_point(c, v) for c, v in zip(selected, vectors)]
    client.upsert("yr", points=points)
    bm25 = BM25Index()
    bm25.build([p.payload for p in points], [str(p.id) for p in points])
    monkeypatch.setattr(settings, "qdrant_collection", "yr")
    monkeypatch.setattr(rm, "get_qdrant_client", lambda: client)
    return rm, bm25


@needs_pdfs
def test_retrieval_uses_current_regulation_when_year_is_a_table_column(chunks, monkeypatch):
    selected = [c for c in chunks["C"] if c.article_id in ("C8.1.8", "C8.1.7", "C8.2.1", "C4.1") and c.is_current]
    selected += [c for c in chunks["B"] if c.article_id in ("B2.5.2", "B9.5.3", "B2.5.3")]
    rm, bm25 = _memory_retriever(monkeypatch, selected)
    res = rm.retrieve("How many times can a team apply for a new ECU in 2028?", bm25_index=bm25, enable_reranker=False)
    assert res.year_resolution.mode == "current" and res.scope.season == 2026
    assert any(c["payload"]["article_id"] == "C8.1.8" and c["payload"]["is_current"] for c in res.candidates)
    assert all(c["payload"]["is_current"] for c in res.candidates)
    res = rm.retrieve("What is the minimum mass of the car?", bm25_index=bm25, enable_reranker=False)
    assert res.year_resolution is None and all(c["payload"]["is_current"] for c in res.candidates)


@needs_pdfs
def test_retrieval_uses_future_provision_when_current_text_never_mentions_the_year(chunks, monkeypatch):
    selected = [c for c in chunks["B"] if c.article_id in ("B2.5.2", "B9.5.3", "B2.5.3", "B2.5.4")]
    rm, bm25 = _memory_retriever(monkeypatch, selected)
    res = rm.retrieve("What is the race distance in 2027?", bm25_index=bm25, enable_reranker=False)
    assert res.year_resolution.mode == "future" and res.scope.season == 2027 and not res.year_resolution.explicit
    assert res.candidates and all(c["payload"]["effective_year"] == 2027 for c in res.candidates)
    res = rm.retrieve("From 2028, how many exceptions will a competitor get in restricted period one?", bm25_index=bm25, enable_reranker=False)
    assert res.year_resolution.mode == "future" and res.year_resolution.explicit
    assert all(c["payload"]["effective_year"] == 2028 for c in res.candidates)
    res = rm.retrieve("What is the race distance?", bm25_index=bm25, enable_reranker=False)
    assert res.year_resolution is None and all(c["payload"]["is_current"] for c in res.candidates)


EVIDENCE = {
    1: {"text": "A speed limit of 80km/h will be imposed in the Pit Lane during the whole Competition. However this limit may be amended by the Race Director.",
        "article_id": "B1.6.3", "issue": 9, "page_start": 10, "page_end": 10, "effective_year": 2026, "is_current": True, "is_binding": True},
    2: {"text": "The Minimum Mass is 726kg plus the Nominal Tyre Mass. In all other sessions, the Minimum Mass is 724kg.",
        "article_id": "C4.1", "issue": 21, "page_start": 59, "page_end": 59, "effective_year": 2026, "is_current": True, "is_binding": True},
    3: {"text": "[driver definitions to be reviewed]", "article_id": "A5.8", "issue": 3, "page_start": 25, "page_end": 25,
        "effective_year": 2026, "is_current": True, "is_binding": False},
    4: {"text": "shall be equal to the least number of complete laps which exceed a distance of 290km", "article_id": "B2.5.2",
        "issue": 9, "page_start": 95, "page_end": 95, "effective_year": 2027, "is_current": False, "is_binding": True},
}


@pytest.mark.parametrize("claim, sources, valid", [
    ("The pit lane speed limit is 80km/h.", [1], True),
    ("The pit lane speed limit is 90km/h.", [1], False),
    ("The pit lane speed limit is 80 km/h.", [1], True),
    ("The pit lane speed limit is 80mph.", [1], False),
    ("The Race Director may amend the pit lane speed limit.", [1], True),
    ("Minimum mass is 726kg plus the Nominal Tyre Mass.", [2], True),
    ("Minimum mass is 726kg plus the Nominal Tyre Mass.", [1], False),
    ("Minimum mass is 724kg in other sessions.", [2], True),
    ("Minimum mass is 740kg in other sessions.", [2], False),
    ("The limit is 80km/h according to Article B1.6.3.", [1], True),
    ("The limit is 80km/h according to Article B1.6.4.", [1], False),
    ("The limit is 80km/h.", [9], False),
    ("Teams are required to wear blue shoes at all times in garages.", [1], False),
])
def test_claim_verification_rules(claim, sources, valid):
    verdict = verify_claim(claim, sources, EVIDENCE, "What is the pit lane limit?")
    assert verdict.valid is valid, verdict.problems


def test_number_must_match_exactly_without_prefix_collisions():
    assert not verify_claim("It is 8km/h.", [1], EVIDENCE, "q").valid
    assert not verify_claim("It is 0km/h.", [1], EVIDENCE, "q").valid
    assert verify_claim("Applies in 2027.", [4], EVIDENCE, "q").valid
    assert not verify_claim("Applies in 2028.", [4], EVIDENCE, "q").valid


def _candidates(*idx):
    return [{"id": f"id{i}", "score": 1.0, "payload": EVIDENCE[i]} for i in idx]


def _scope():
    return QueryScope(season=2026)


def test_answer_contains_backend_markers_and_resolved_citations():
    reply = {"can_answer": True, "claims": [{"text": "The pit lane speed limit is 80km/h.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("What is the pit lane limit?", _candidates(1, 2), _scope())
    assert not r.abstained and r.answer.endswith("[C1]")
    assert [c.ref for c in r.citations] == ["C1"]
    c = r.citations[0]
    assert (c.section, c.article_id, c.issue, c.page_start, c.is_current) == ("", "B1.6.3", 9, 10, True) or c.article_id == "B1.6.3"
    assert c.issue == 9 and c.page_start == 10


def test_invented_citation_identifiers_cannot_reach_the_answer():
    reply = {"can_answer": True, "claims": [{"text": "The pit lane speed limit is 80km/h [C9] per Article Z9.9.", "sources": [9]}]}
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "no_verified_claims" and r.citations == []
    assert r.answer.startswith(INSUFFICIENT_EVIDENCE)


def test_fabricated_number_is_removed_and_valid_claim_is_kept():
    reply = {"can_answer": True, "claims": [
        {"text": "The pit lane speed limit is 80km/h.", "sources": [1]},
        {"text": "Speeding costs a fine of 5000 euros.", "sources": [1]},
    ]}
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(1), _scope())
    assert "5000" not in r.answer and "80km/h" in r.answer
    assert any("removed" in w for w in r.warnings) and "omitted" in r.answer


def test_claim_is_recited_to_the_chunk_that_actually_supports_it():
    reply = {"can_answer": True, "claims": [{"text": "The Minimum Mass is 726kg plus the Nominal Tyre Mass.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(1, 2), _scope())
    assert not r.abstained and [c.ref for c in r.citations] == ["C2"]
    assert any("corrected" in w for w in r.warnings)


def test_non_binding_comment_is_labelled_and_never_presented_as_requirement():
    reply = {"can_answer": True, "claims": [{"text": "Driver definitions are marked to be reviewed.", "sources": [3]}]}
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(3), _scope())
    assert r.answer.startswith("Non-binding comment:") and r.citations[0].is_binding is False


def test_future_provisions_are_labelled_not_in_force():
    reply = {"can_answer": True, "claims": [{"text": "The race distance exceeds 290km.", "sources": [4]}]}
    res = YearResolution("future", 2027, "explicit", explicit=True)
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(4), QueryScope(season=2027), year_resolution=res)
    assert "approved future change effective 2027, not currently in force" in r.answer
    assert r.citations[0].is_current is False and r.year_interpretation["mode"] == "future"


def test_exact_lookup_truncation_is_exposed():
    reply = {"can_answer": True, "claims": [{"text": "The pit lane speed limit is 80km/h.", "sources": [1]}]}
    with patch(GEN, return_value=reply) as mocked:
        r = generate("q", _candidates(1), _scope(), exact_total_chunks=119, exact_truncated=True)
    assert r.context_truncated and "1 of 119" in r.answer and any("truncated" in w.lower() for w in r.warnings)
    assert "1 of 119" in mocked.call_args[0][1]


def test_abstains_without_evidence_when_model_declines_and_when_generation_fails():
    with patch(GEN) as mocked:
        r = generate("q", [], _scope())
        assert r.abstained and r.abstain_reason == "no_evidence" and not mocked.called
    with patch(GEN, return_value={"can_answer": False, "claims": []}):
        r = generate("q", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "model_declined" and r.citations == []
    with patch(GEN, side_effect=RuntimeError("LLM unavailable")):
        r = generate("q", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "generation_failed" and r.answer.startswith("Generation failed")


def test_past_year_without_evidence_states_that_only_2026_is_indexed():
    res = YearResolution("past", 2023, "only 2026 regulations are indexed")
    r = generate("q", [], QueryScope(season=2023), year_resolution=res)
    assert r.abstained and "2026 season only" in r.answer


def test_prompt_marks_future_and_non_binding_evidence_and_year_notes():
    block = build_context_block([{"payload": {**EVIDENCE[3], "section_code": "A"}}, {"payload": {**EVIDENCE[4], "section_code": "B"}}])
    assert "NON-BINDING COMMENT" in block and "FUTURE CHANGE effective 2027, NOT in force" in block
    msg = build_user_message("q", _candidates(1), ["note one"])
    assert "note one" in msg and "[C1]" in msg and "p10" in msg and "Issue 9" in msg


@pytest.mark.parametrize("question, parts", [
    ("How many wet tyre sets can a driver use, and what is the pit lane speed limit?", 2),
    ("Can a car be disqualified for being too light, and what is the minimum mass?", 2),
    ("Who decides cost cap cases and when are documents due?", 2),
    ("What are the rules for pit stops and tyres?", 1),
    ("What is the speed limit in the pit lane?", 1),
    ("Compare gearbox and clutch requirements", 1),
])
def test_multi_part_questions_are_split_only_at_a_second_question(question, parts):
    from app.retrieval.query_processor import split_question

    assert len(split_question(question)) == parts


def test_multi_part_retrieval_merges_evidence_from_each_part(monkeypatch):
    from app.retrieval import retriever as rm

    def fake(part, bm25_index, **kwargs):
        ids = ["a1", "a2", "shared"] if "tyre" in part else ["b1", "shared", "b2"]
        scope = QueryScope(season=2026)
        return rm.RetrievalResult([{"id": i, "score": 1.0, "payload": {}} for i in ids], scope, False, {"total_ms": 5.0})

    monkeypatch.setattr(rm, "retrieve", fake)
    res = rm.retrieve_multi("How many tyre sets are allowed, and what is the speed limit?", bm25_index=None)
    ids = [c["id"] for c in res.candidates]
    assert ids == ["a1", "b1", "a2", "shared", "b2"] and res.latency_ms["total_ms"] == 10.0
    single = rm.retrieve_multi("What is the speed limit?", bm25_index=None)
    assert [c["id"] for c in single.candidates] == ["b1", "shared", "b2"]


def test_apostrophe_before_a_number_is_not_treated_as_a_thousands_separator():
    ev = {1: {"text": "Pounds Sterling (‘000): 152,918 | Euros (‘000): 180,420", "article_id": "E2.3.2", "issue": 7, "page_start": 6, "page_end": 6}}
    assert verify_claim("The cap is 152,918 ('000).", [1], ev, "q").valid
    assert not verify_claim("The cap is 152,919 ('000).", [1], ev, "q").valid
    assert not verify_claim("The cap is 52,918.", [1], ev, "q").valid


def test_status_only_claims_are_dropped_and_padding_citations_are_pruned():
    reply = {"can_answer": True, "claims": [
        {"text": "The race distance exceeds 290km.", "sources": [4, 1, 2]},
        {"text": "This regulation is an approved future change and is not currently in force.", "sources": [4]},
    ]}
    with patch(GEN, return_value=reply):
        r = generate("q", _candidates(4, 1, 2), QueryScope(season=2027))
    assert [c.ref for c in r.citations] == ["C1"]
    assert len(r.claims) == 1 and "not currently in force" in r.answer


def test_prompt_is_shortened_to_fit_the_context_window(monkeypatch):
    big = {"text": "word " * 3000, "article_id": "B1.1", "issue": 9, "page_start": 1, "page_end": 1, "is_current": True}
    cands = [{"id": f"i{i}", "score": 1.0, "payload": {**big, "article_id": f"B1.{i}"}} for i in range(1, 6)]
    reply = {"can_answer": False, "claims": []}
    with patch(GEN, return_value=reply) as mocked:
        r = generate("q", cands, _scope())
    assert mocked.call_args[0][1].count("[C") < 5
    assert any("shortened" in w for w in r.warnings)


def test_ordinals_in_a_claim_must_appear_in_the_cited_source():
    prose = {"text": "If the leader has completed 75% or more of the distance, points are awarded in accordance with column 4 of the table, 2 points for two laps.",
             "article_id": "A2.2.1", "issue": 3, "page_start": 9, "page_end": 9}
    table = {"text": "Position: 9th | Points Col. 4 ≥75%: 2", "article_id": "A2.2.1", "issue": 3, "page_start": 10, "page_end": 10}
    claim = "A driver finishing 9th receives 2 points."
    assert not verify_claim(claim, [1], {1: prose, 2: table}, "q").valid
    assert verify_claim(claim, [2], {1: prose, 2: table}, "q").valid
    reply = {"can_answer": True, "claims": [{"text": claim, "sources": [1]}]}
    cands = [{"id": "p", "score": 1.0, "payload": prose}, {"id": "t", "score": 1.0, "payload": table}]
    with patch(GEN, return_value=reply):
        r = generate("What does 9th place get?", cands, _scope())
    assert [c.ref for c in r.citations] == ["C2"] and r.citations[0].page_start == 10


def test_abstention_on_a_truncated_exact_lookup_explains_the_truncation():
    with patch(GEN, return_value={"can_answer": False, "claims": []}):
        r = generate("What does Article B5 say?", _candidates(1), _scope(), exact_total_chunks=119, exact_truncated=True)
    assert r.abstained and r.context_truncated
    assert "1 of 119" in r.answer and "specific clause" in r.answer


def test_letter_number_codes_are_not_read_as_quantities_with_units():
    ev = {1: {"text": "Drivers eliminated in Q1 who set a lap slower than the limit start behind.", "article_id": "B2.4.1", "issue": 9, "page_start": 1, "page_end": 1}}
    assert verify_claim("Drivers eliminated in Q1 who set a slow Q1 lap start behind.", [1], ev, "q").valid


def test_quantity_gate_is_skipped_for_multi_part_questions():
    reply = {"asks": "x", "not_covered": "rim diameter", "claims": [{"text": "The Race Director may amend this limit.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("What is the rim diameter, and who may amend the limit?", _candidates(1), _scope())
    assert not r.abstained


def test_statements_about_what_the_regulations_do_not_say_are_removed_and_reported():
    reply = {"asks": "limit", "not_covered": "the number of wet tyre sets", "claims": [
        {"text": "The regulations do not state a specific maximum number of sets.", "sources": [1]},
        {"text": "The pit lane speed limit is 80km/h.", "sources": [1]},
    ]}
    with patch(GEN, return_value=reply):
        r = generate("What is the pit lane speed limit?", _candidates(1), _scope())
    assert "do not state" not in r.answer and "80km/h" in r.answer
    assert "Not covered by the available evidence: the number of wet tyre sets." in r.answer
    assert any("absence" in w for w in r.warnings)


def test_quantity_questions_need_a_verified_quantity():
    reply = {"asks": "limits", "not_covered": "", "claims": [{"text": "The Race Director may amend this limit.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("What is the maximum speed in the pit lane?", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "no_quantity_in_evidence"
    with patch(GEN, return_value=reply):
        r = generate("What may the Race Director do about the limit?", _candidates(1), _scope())
    assert not r.abstained


def test_answers_without_a_can_answer_flag_are_accepted_and_empty_claims_abstain():
    with patch(GEN, return_value={"asks": "x", "not_covered": "", "claims": []}):
        r = generate("q", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "model_declined"


def test_equivalent_time_and_number_formats_are_accepted_only_when_the_value_matches():
    ev = {1: {"text": "The deadline is 19.00 CET on 31 March, with 2000 runs and 152,918 units.", "article_id": "D1.1", "issue": 8, "page_start": 1, "page_end": 1}}
    assert verify_claim("Submit by 19:00 CET on 31 March.", [1], ev, "q").valid
    assert not verify_claim("Submit by 19:30 CET on 31 March.", [1], ev, "q").valid
    assert verify_claim("There are 2,000 runs and 152’918 units.", [1], ev, "q").valid
    assert not verify_claim("There are 2,001 runs.", [1], ev, "q").valid
    assert not verify_claim("There are 152.918 units.", [1], ev, "q").valid


def test_status_only_claims_about_comments_are_dropped():
    reply = {"asks": "note", "not_covered": "", "claims": [
        {"text": "Article D1.3.1 is a non-binding comment and not a regulatory requirement.", "sources": [3]},
        {"text": "The current content of D1.3.1 is a non-binding comment.", "sources": [3]},
        {"text": "Driver definitions are to be reviewed.", "sources": [3]},
    ]}
    with patch(GEN, return_value=reply):
        r = generate("What does the note say?", _candidates(3), _scope())
    assert r.answer.count("Non-binding comment:") == 1 and "to be reviewed" in r.answer


def test_article_completion_adds_the_best_matching_sibling_chunk_only(chunks_unused=None):
    import hashlib

    from qdrant_client import QdrantClient

    from app.models.schemas import Chunk
    from app.retrieval.retriever import complete_articles
    from app.storage.qdrant_store import chunk_to_point, ensure_collection

    def mk(aid, text, i):
        return Chunk(chunk_id=hashlib.md5(f"{aid}{i}".encode()).hexdigest(), document_id="d", season=2026, section_code="D",
                     section_name="x", issue=8, article_id=aid, article_title=None, hierarchy_path="D", page_start=8, page_end=8,
                     content_type="article", content_role="regulatory_text", source_url="u", text=text, embedding_text=text,
                     article_path=[aid], effective_year=2026, is_current=True)

    a, b, c = mk("D4.1.3", "prose", 1), mk("D4.1.3", "table euros", 2), mk("D4.1.3", "other", 3)
    client = QdrantClient(":memory:")
    ensure_collection(client, "t", 4)
    client.upsert("t", points=[chunk_to_point(a, [1, 0, 0, 0]), chunk_to_point(b, [0, 1, 0, 0]), chunk_to_point(c, [0, 0, 1, 0])])
    settings_collection = settings.qdrant_collection
    settings.qdrant_collection = "t"
    try:
        top = [{"id": str(__import__("uuid").UUID(a.chunk_id)), "score": 1.0, "payload": {"article_id": "D4.1.3"}}]
        out = complete_articles(top, client, [0.0, 1.0, 0.0, 0.0], 2026, "D")
    finally:
        settings.qdrant_collection = settings_collection
    assert len(out) == 2 and out[1]["payload"]["text"] == "table euros" and "vector" not in out[1]


def test_single_part_question_abstains_when_model_reports_requested_information_missing():
    reply = {"asks": "top speed", "not_covered": "The maximum top speed a car can reach", "claims": [
        {"text": "The Race Director may amend this limit.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("What is the top speed a car can reach?", _candidates(1), _scope())
    assert r.abstained and r.abstain_reason == "requested_information_missing"
    assert "Not covered by the available evidence" in r.answer and "top speed" in r.answer


def test_minor_not_covered_note_does_not_abstain_single_part_answers():
    reply = {"asks": "limit", "not_covered": "the exact fine amounts", "claims": [
        {"text": "The Race Director may amend this limit.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("What may the Race Director do about the pit lane limit?", _candidates(1), _scope())
    assert not r.abstained and "exact fine amounts" in r.answer


def test_unanswered_part_of_multi_part_question_is_named_even_if_model_stays_silent():
    reply = {"asks": "x", "not_covered": "", "claims": [{"text": "The Race Director may amend this limit.", "sources": [1]}]}
    with patch(GEN, return_value=reply):
        r = generate("How many wet-weather tyre sets can a driver use, and who may amend the limit?", _candidates(1), _scope())
    assert not r.abstained
    assert "Not covered by the available evidence: How many wet-weather tyre sets can a driver use" in r.answer
    assert "amend" not in r.answer.split("Not covered")[1]
