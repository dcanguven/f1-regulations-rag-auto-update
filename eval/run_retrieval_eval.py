from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.config import settings
from app.retrieval import retriever as retriever_module
from app.retrieval.embedder import Embedder
from app.retrieval.query_processor import extract_scope
from app.retrieval.retriever import retrieve
from app.retrieval.rrf import rrf_fuse
from app.retrieval.sparse import BM25Index, build_bm25_from_qdrant
from app.storage.qdrant_store import get_qdrant_client, scroll_by_article, search_dense

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "retrieval_eval_v1.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)
METHODS = ("dense", "bm25", "hybrid", "pipeline", "dense_nofilter", "bm25_nofilter", "hybrid_nofilter")
LATENCY_RUNS = 3


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_relevant(payload: dict, q: dict) -> bool:
    if payload.get("document_id") != q["document_id"] or payload.get("effective_year") != q["effective_year"]:
        return False
    aid = payload.get("article_id")
    return bool(aid) and any(aid == r or aid.startswith(r + ".") for r in q["relevant_article_ids"])


def has_evidence(payload: dict, q: dict) -> bool:
    if payload.get("document_id") != q["document_id"] or payload.get("effective_year") != q["effective_year"]:
        return False
    text = norm(payload.get("text", ""))
    return all(norm(e["text"]) in text for e in q["evidence"])


def first_rank(flags: list[bool]) -> int | None:
    for i, f in enumerate(flags, start=1):
        if f:
            return i
    return None


def scores_for(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    out["mrr@10"] = sum((1 / r) for r in ranks if r is not None and r <= 10) / n
    out["n"] = n
    return out


def run_methods(q, scope, embedder, qdrant, bm25):
    vec = embedder.encode_query(q["question"])
    kw = dict(section_filter=scope.section_code, effective_year_filter=scope.season)
    dense = search_dense(qdrant, settings.qdrant_collection, vec, settings.dense_top_k, **kw)
    sparse = bm25.search(
        q["question"], settings.sparse_top_k, section_filter=scope.section_code, effective_year_filter=scope.season
    )
    hybrid = rrf_fuse(dense, sparse, settings.rrf_top_k)
    kw0 = dict(section_filter=None, effective_year_filter=scope.season)
    dense0 = search_dense(qdrant, settings.qdrant_collection, vec, settings.dense_top_k, **kw0)
    sparse0 = bm25.search(q["question"], settings.sparse_top_k, **kw0)
    return {
        "dense": dense, "bm25": sparse, "hybrid": hybrid,
        "dense_nofilter": dense0, "bm25_nofilter": sparse0,
        "hybrid_nofilter": rrf_fuse(dense0, sparse0, settings.rrf_top_k),
    }


def main() -> None:
    questions = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    qdrant = get_qdrant_client()
    retriever_module.get_qdrant_client = lambda: qdrant
    embedder = Embedder()
    t0 = time.perf_counter()
    embedder._load()
    init_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    bm25 = build_bm25_from_qdrant(qdrant, settings.qdrant_collection)
    bm25_build_ms = (time.perf_counter() - t0) * 1000
    for _ in range(3):
        embedder.encode_query("warm up query about regulations")

    details = []
    ranks = {m: {"article": {}, "evidence": {}} for m in METHODS}
    exact_rows = []
    scope_rows = []
    contamination = []
    abstain = {"answerable": [], "unanswerable": []}

    for q in questions:
        scope = extract_scope(q["question"])
        scope_rows.append(
            {"id": q["id"], "exact_article": scope.exact_article, "section_code": scope.section_code,
             "season": scope.season, "explicit": q.get("explicit_reference", False), "answerable": q["answerable"]}
        )
        results = run_methods(q, scope, embedder, qdrant, bm25)
        pipe = retrieve(q["question"], bm25_index=bm25, enable_reranker=False)
        results["pipeline"] = pipe.candidates
        row = {"id": q["id"], "category": q["category"], "language": q["language"], "question": q["question"],
               "scope": {"exact_article": scope.exact_article, "section": scope.section_code, "season": scope.season},
               "was_exact_lookup": pipe.was_exact_lookup, "methods": {}}
        for m in METHODS:
            cands = results[m]
            row["methods"][m] = [
                {"id": str(c.get("id")), "article": c["payload"].get("article_id"), "page": c["payload"].get("page_start"),
                 "year": c["payload"].get("effective_year"), "role": c["payload"].get("content_role"),
                 "doc": c["payload"].get("document_id"), "score": round(float(c.get("score", 0)), 4)}
                for c in cands[:10]
            ]
            for c in cands[:10]:
                for ev in q.get("forbidden_evidence", []):
                    if ev in c["payload"].get("text", "") and c["payload"].get("effective_year") != q["effective_year"]:
                        contamination.append({"id": q["id"], "method": m, "forbidden": ev, "year": c["payload"].get("effective_year")})
            if q["answerable"]:
                ranks[m]["article"][q["id"]] = first_rank([is_relevant(c["payload"], q) for c in cands[:10]])
                ranks[m]["evidence"][q["id"]] = first_rank([has_evidence(c["payload"], q) for c in cands[:10]])
        if pipe.was_exact_lookup and q["answerable"]:
            exact_rows.append(
                {"id": q["id"], "returned_chunks": len(pipe.candidates),
                 "target_present": any(is_relevant(c["payload"], q) for c in pipe.candidates),
                 "evidence_present": any(has_evidence(c["payload"], q) for c in pipe.candidates)}
            )
        top = results["hybrid"][:1]
        dtop = results["dense"][:1]
        (abstain["answerable"] if q["answerable"] else abstain["unanswerable"]).append(
            {"id": q["id"], "dense_top1": float(dtop[0]["score"]) if dtop else 0.0, "rrf_top1": float(top[0]["score"]) if top else 0.0,
             "bm25_top1": float(results["bm25"][0]["score"]) if results["bm25"] else 0.0}
        )
        details.append(row)

    latency = defaultdict(list)
    for _ in range(LATENCY_RUNS):
        for q in questions:
            scope = extract_scope(q["question"])
            t = time.perf_counter()
            vec = embedder.encode_query(q["question"])
            t_embed = (time.perf_counter() - t) * 1000
            kw = dict(section_filter=scope.section_code, effective_year_filter=scope.season)
            t = time.perf_counter()
            dense = search_dense(qdrant, settings.qdrant_collection, vec, settings.dense_top_k, **kw)
            t_dense = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            sparse = bm25.search(q["question"], settings.sparse_top_k, **kw)
            t_bm25 = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            rrf_fuse(dense, sparse, settings.rrf_top_k)
            t_rrf = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            pipe = retrieve(q["question"], bm25_index=bm25, enable_reranker=False)
            t_pipe = (time.perf_counter() - t) * 1000
            latency["query_embed"].append(t_embed)
            latency["dense_search"].append(t_dense)
            latency["bm25_search"].append(t_bm25)
            latency["rrf"].append(t_rrf)
            latency["dense_total"].append(t_embed + t_dense)
            latency["hybrid_total"].append(t_embed + t_dense + t_bm25 + t_rrf)
            latency["pipeline_total_exact" if pipe.was_exact_lookup else "pipeline_total_hybrid"].append(t_pipe)

    answerable = [q for q in questions if q["answerable"]]
    by = {}
    for m in METHODS:
        for kind in ("article", "evidence"):
            by[f"{m}/{kind}"] = {
                "overall": scores_for([ranks[m][kind][q["id"]] for q in answerable]),
                "by_category": {},
                "by_language": {},
            }
            cats = defaultdict(list)
            langs = defaultdict(list)
            for q in answerable:
                cats[q["category"]].append(ranks[m][kind][q["id"]])
                langs[q["language"]].append(ranks[m][kind][q["id"]])
            by[f"{m}/{kind}"]["by_category"] = {c: scores_for(v) for c, v in cats.items()}
            by[f"{m}/{kind}"]["by_language"] = {c: scores_for(v) for c, v in langs.items()}

    summary = {
        "collection": settings.qdrant_collection,
        "questions": len(questions),
        "answerable": len(answerable),
        "reranker_enabled": False,
        "model_init_ms": round(init_ms, 1),
        "bm25_build_ms": round(bm25_build_ms, 1),
        "bm25_documents": len(bm25._payloads),
        "metrics": by,
        "latency_ms_median": {k: round(statistics.median(v), 2) for k, v in latency.items()},
        "latency_ms_p95": {k: round(sorted(v)[int(0.95 * (len(v) - 1))], 2) for k, v in latency.items()},
        "latency_samples": {k: len(v) for k, v in latency.items()},
        "exact_lookup_rows": exact_rows,
        "scope_rows": scope_rows,
        "abstention_signals": abstain,
        "future_contamination": contamination,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_v1.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (RESULTS / "baseline_v1_details.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details))
    qdrant.close()
    print("done", len(questions), "questions")


if __name__ == "__main__":
    sys.exit(main())
