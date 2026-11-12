from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.generation.answer import MAX_EVIDENCE, generate
from app.generation.verification import normalise
from app.retrieval import retriever as retriever_module
from app.retrieval.retriever import retrieve_multi
from app.retrieval.sparse import build_bm25_from_qdrant
from app.storage.qdrant_store import get_qdrant_client
from run_retrieval_eval import norm

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / (sys.argv[1] if len(sys.argv) > 1 else "answer_eval_v1.jsonl")
RESULTS = ROOT / "eval" / "results"


def matches(payload: dict, cand: dict) -> bool:
    if payload.get("document_id") != cand["document_id"] or payload.get("effective_year") != cand["effective_year"]:
        return False
    aid = payload.get("article_id")
    return bool(aid) and any(aid == r or aid.startswith(r + ".") for r in cand["relevant_article_ids"])


def has_evidence(payload: dict, cand: dict) -> bool:
    if not cand["evidence"]:
        return False
    text = norm(payload.get("text", ""))
    return all(norm(e["text"]) in text for e in cand["evidence"])


def score(q: dict, resp, evidence: dict[int, dict]) -> dict:
    out: dict = {}
    answer = resp.answer
    answer_norm = normalise(answer)
    cited = {c.ref: evidence[int(c.ref[1:])] for c in resp.citations if int(c.ref[1:]) in evidence}
    out["abstained"] = resp.abstained
    out["abstain_reason"] = resp.abstain_reason
    if not q["answerable"]:
        out["abstention_correct"] = resp.abstained
        out["correct"] = resp.abstained
        return out
    if resp.abstained:
        out["abstention_correct"] = bool(q.get("abstain_acceptable"))
        out["correct"] = bool(q.get("abstain_acceptable"))
        out["citation_precision"] = None
        return out
    out["abstention_correct"] = True
    must_ok = all(normalise(m) in answer_norm for m in q["must_include"])
    any_ok = all(any(normalise(m) in answer_norm for m in group) for group in q["must_include_any"])
    forbidden = [m for m in q["must_not_include"] if normalise(m) in answer_norm]
    out["facts_present"] = must_ok and any_ok
    out["forbidden_present"] = forbidden

    groups = q["groups"]
    group_hit, group_evidence = [], []
    for g in groups:
        group_hit.append(any(matches(p, c) for p in cited.values() for c in g["candidates"]))
        group_evidence.append(any(has_evidence(p, c) for p in cited.values() for c in g["candidates"]))
    out["groups_cited"] = sum(group_hit) / len(groups)
    out["groups_with_evidence_chunk"] = sum(group_evidence) / len(groups)
    all_cands = [c for g in groups for c in g["candidates"]]
    correct_cites, page_ok, url_ok = 0, 0, 0
    for c in resp.citations:
        payload = evidence[int(c.ref[1:])]
        hit = next((cd for cd in all_cands if matches(payload, cd)), None)
        correct_cites += hit is not None
        if hit is not None and hit["evidence"]:
            pages = [e["page"] for e in hit["evidence"]]
            page_ok += any(c.page_start <= pg <= c.page_end for pg in pages)
        elif hit is not None:
            page_ok += 1
        url_ok += bool(c.source_url) and f"iss_{c.issue:02d}" in c.source_url.lower()
    n = max(len(resp.citations), 1)
    out["citation_precision"] = correct_cites / n
    out["citation_page_accuracy"] = page_ok / n
    out["citation_url_ok"] = url_ok / n
    markers = set(re.findall(r"\[C(\d+)\]", answer))
    out["marker_integrity"] = markers == {c.ref[1:] for c in resp.citations}

    status = q["year_status"]
    future_cited = [c for c in resp.citations if not c.is_current]
    if status == "current":
        out["year_correct"] = not future_cited and "future change" not in answer_norm
    elif status == "future":
        out["year_correct"] = bool(future_cited) and "approved future change" in answer_norm and "not currently in force" in answer_norm
    else:
        out["year_correct"] = None
    out["binding_correct"] = (not resp.citations or all(not c.is_binding for c in resp.citations)) and answer_norm.startswith("non-binding comment") if not q["expect_binding"] else all(c.is_binding for c in resp.citations)
    if q["category"] == "ambiguous":
        out["correct"] = not forbidden and out["groups_cited"] > 0 and out["citation_precision"] == 1.0
        docs = {c.document_id for c in resp.citations}
        out["ambiguity_presented_both"] = len(docs) > 1
    else:
        out["correct"] = (
            out["facts_present"] and not forbidden and out["citation_precision"] == 1.0
            and out["groups_cited"] == 1.0 and out["year_correct"] is not False and out["binding_correct"]
        )
    return out


def main() -> None:
    questions = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    client = get_qdrant_client()
    retriever_module.get_qdrant_client = lambda: client
    bm25 = build_bm25_from_qdrant(client, settings.qdrant_collection)
    records = []
    for q in questions:
        t0 = time.perf_counter()
        res = retrieve_multi(q["question"], bm25_index=bm25, enable_reranker=False)
        t1 = time.perf_counter()
        resp = generate(
            q["question"], res.candidates, res.scope, debug=True, year_resolution=res.year_resolution,
            exact_total_chunks=res.exact_total_chunks, exact_truncated=res.exact_truncated,
        )
        t2 = time.perf_counter()
        usable = [c for c in res.candidates if c["payload"].get("text", "").strip()][:MAX_EVIDENCE]
        evidence = {i: c["payload"] for i, c in enumerate(usable, start=1)}
        rec = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "answer": resp.answer, "abstained": resp.abstained, "abstain_reason": resp.abstain_reason,
            "citations": [vars(c) for c in resp.citations],
            "claims": [vars(c) for c in resp.claims], "warnings": resp.warnings,
            "year_interpretation": resp.year_interpretation,
            "retrieved": [(c["payload"].get("article_id"), c["payload"].get("page_start"), c["payload"].get("effective_year")) for c in usable],
            "retrieval_ms": round((t1 - t0) * 1000, 1), "generation_ms": round((t2 - t1) * 1000, 1),
            "llm": (resp.debug or {}).get("llm"), "rejected": (resp.debug or {}).get("rejected"),
            "scores": score(q, resp, evidence),
        }
        records.append(rec)
        print(f"{q['id']} {q['category']:15} correct={rec['scores']['correct']} abstained={resp.abstained} gen={rec['generation_ms']:.0f}ms", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / (sys.argv[2] if len(sys.argv) > 2 else "answer_eval.json")).write_text(json.dumps(records, indent=1, ensure_ascii=False))
    gens = [r["generation_ms"] for r in records if r["generation_ms"] > 50]
    print("generation ms: mean %.0f median %.0f p95 %.0f" % (statistics.mean(gens), statistics.median(gens), sorted(gens)[int(0.95 * (len(gens) - 1))]))
    client.close()


if __name__ == "__main__":
    main()
