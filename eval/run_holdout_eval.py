from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.retrieval import retriever as retriever_module
from app.retrieval.embedder import Embedder
from app.retrieval.query_processor import extract_scope
from app.retrieval.retriever import retrieve, validate_scope
from app.retrieval.rrf import rrf_fuse
from app.retrieval.sparse import build_bm25_from_qdrant
from app.storage.qdrant_store import get_qdrant_client, search_dense
from experiment_lib import first_rank
from run_retrieval_eval import norm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)
METHODS = ("dense", "bm25", "hybrid", "pipeline")
LATENCY_RUNS = 3


def from_flat(q: dict) -> dict:
    if not q["answerable"]:
        return {**q, "groups": []}
    cand = {
        "document_id": q["document_id"],
        "relevant_article_ids": q["relevant_article_ids"],
        "effective_year": q["effective_year"],
        "evidence": q["evidence"],
    }
    return {**q, "groups": [{"candidates": [cand]}]}


def load(path: Path, flat: bool) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [from_flat(r) for r in rows] if flat else rows


def cand_relevant(payload: dict, cand: dict) -> bool:
    if payload.get("document_id") != cand["document_id"] or payload.get("effective_year") != cand["effective_year"]:
        return False
    aid = payload.get("article_id")
    return bool(aid) and any(aid == r or aid.startswith(r + ".") for r in cand["relevant_article_ids"])


def cand_evidence(payload: dict, cand: dict) -> bool:
    if not cand["evidence"]:
        return False
    if payload.get("document_id") != cand["document_id"] or payload.get("effective_year") != cand["effective_year"]:
        return False
    text = norm(payload.get("text", ""))
    return all(norm(e["text"]) in text for e in cand["evidence"])


def any_group(payload: dict, q: dict, fn) -> bool:
    return any(fn(payload, c) for g in q["groups"] for c in g["candidates"])


def has_evidence_labels(q: dict) -> bool:
    return any(c["evidence"] for g in q["groups"] for c in g["candidates"])


def coverage(cands: list[dict], q: dict, depth: int) -> float:
    groups = q["groups"]
    hit = sum(1 for g in groups if any(cand_relevant(c["payload"], cd) for c in cands[:depth] for cd in g["candidates"]))
    return hit / len(groups)


def summarise(ranks: list[tuple[int | None, int | None, bool]]) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        out[f"art@{k}"] = sum(1 for a, _, _ in ranks if a is not None and a <= k) / n
    out["mrr@10"] = sum(1 / a for a, _, _ in ranks if a is not None and a <= 10) / n
    labelled = [(e) for _, e, h in ranks if h]
    out["ev_n"] = len(labelled)
    for k in (1, 3, 5, 10):
        out[f"ev@{k}"] = (sum(1 for e in labelled if e is not None and e <= k) / len(labelled)) if labelled else None
    return out


def run(dataset: Path, flat: bool, tag: str) -> dict:
    questions = load(dataset, flat)
    qdrant = get_qdrant_client()
    retriever_module.get_qdrant_client = lambda: qdrant
    embedder = Embedder()
    embedder.encode_query("warm up")
    bm25 = build_bm25_from_qdrant(qdrant, settings.qdrant_collection)
    for _ in range(3):
        embedder.encode_query("warm up query about regulations")

    per_method = {m: {} for m in METHODS}
    coverage5 = {m: {} for m in METHODS}
    details = []
    contamination = 0
    scores_top1 = {"answerable": [], "unanswerable": []}
    for q in questions:
        scope = validate_scope(extract_scope(q["question"]), qdrant)
        vec = embedder.encode_query(q["question"])
        kw = dict(section_filter=scope.section_code, effective_year_filter=scope.season)
        dense = search_dense(qdrant, settings.qdrant_collection, vec, settings.dense_top_k, **kw)
        sparse = bm25.search(q["question"], settings.sparse_top_k, **kw)
        lists = {"dense": dense, "bm25": sparse, "hybrid": rrf_fuse(dense, sparse, settings.rrf_top_k)}
        pipe = retrieve(q["question"], bm25_index=bm25, enable_reranker=False)
        lists["pipeline"] = pipe.candidates
        row = {"id": q["id"], "category": q["category"], "question": q["question"], "was_exact": pipe.was_exact_lookup,
               "scope": [scope.exact_article, scope.section_code, scope.season], "top": {}}
        for m in METHODS:
            cands = lists[m][:10]
            row["top"][m] = [(c["payload"].get("article_id"), c["payload"].get("page_start"), c["payload"].get("effective_year")) for c in cands[:5]]
            if q["answerable"]:
                a = first_rank([any_group(c["payload"], q, cand_relevant) for c in cands])
                e = first_rank([any_group(c["payload"], q, cand_evidence) for c in cands])
                per_method[m][q["id"]] = (a, e, has_evidence_labels(q))
                coverage5[m][q["id"]] = coverage(cands, q, 5)
                for c in cands:
                    year = {cd["effective_year"] for g in q["groups"] for cd in g["candidates"]}
                    if c["payload"].get("effective_year") not in year and not (len(year) > 1):
                        contamination += 1
        (scores_top1["answerable" if q["answerable"] else "unanswerable"]).append(
            {"id": q["id"], "dense_top1": float(dense[0]["score"]) if dense else 0.0, "rrf_top1": float(lists["hybrid"][0]["score"]) if lists["hybrid"] else 0.0}
        )
        details.append(row)

    latency = defaultdict(list)
    for _ in range(LATENCY_RUNS):
        for q in questions:
            scope = validate_scope(extract_scope(q["question"]), qdrant)
            kw = dict(section_filter=scope.section_code, effective_year_filter=scope.season)
            t = time.perf_counter()
            vec = embedder.encode_query(q["question"])
            latency["embed"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            dense = search_dense(qdrant, settings.qdrant_collection, vec, settings.dense_top_k, **kw)
            latency["dense_search"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            bm25.search(q["question"], settings.sparse_top_k, **kw)
            latency["bm25_search"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            pipe = retrieve(q["question"], bm25_index=bm25, enable_reranker=False)
            latency["pipeline_exact" if pipe.was_exact_lookup else "pipeline_hybrid"].append((time.perf_counter() - t) * 1000)

    cats = defaultdict(list)
    for q in questions:
        if q["answerable"]:
            cats[q["category"]].append(q["id"])
    result = {
        "dataset": str(dataset.name), "tag": tag, "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "overall": {m: summarise(list(per_method[m].values())) for m in METHODS},
        "by_category": {m: {c: summarise([per_method[m][i] for i in ids]) for c, ids in cats.items()} for m in METHODS},
        "coverage_at_5": {m: {c: statistics.mean(coverage5[m][i] for i in ids) for c, ids in cats.items()} for m in METHODS},
        "latency_ms": {k: {"median": round(statistics.median(v), 2), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 2), "n": len(v)} for k, v in latency.items()},
        "scores_top1": scores_top1,
        "future_year_mismatches_single_year_questions": contamination,
        "per_question": {m: {i: list(v) for i, v in per_method[m].items()} for m in METHODS},
        "details": details,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    qdrant.close()
    return result


def show(r: dict) -> None:
    def fmt(d):
        ev = "n/a" if d["ev@5"] is None else f"{d['ev@1']:.3f}/{d['ev@5']:.3f}"
        return f"art R@1={d['art@1']:.3f} R@3={d['art@3']:.3f} R@5={d['art@5']:.3f} R@10={d['art@10']:.3f} MRR={d['mrr@10']:.3f} | ev R@1/R@5={ev} (n={d['ev_n']})"
    print(f"== {r['tag']}: {r['questions']} questions, {r['answerable']} answerable")
    for m in METHODS:
        print(f"  {m:9} {fmt(r['overall'][m])}")
    print("  by category (pipeline):")
    for c, d in r["by_category"]["pipeline"].items():
        print(f"    {c:18} n={d['n']} {fmt(d)} | coverage@5={r['coverage_at_5']['pipeline'][c]:.2f}")
    print("  latency:", r["latency_ms"])


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        show(run(ROOT / "eval" / "retrieval_eval_v1.jsonl", True, "selftest_v1_groupschema"))
    else:
        show(run(ROOT / "eval" / sys.argv[1], False, sys.argv[2]))
