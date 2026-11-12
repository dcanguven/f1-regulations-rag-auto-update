from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_bm25 import BM25Okapi

from app.config import settings
from app.retrieval import retriever as retriever_module
from app.retrieval.embedder import Embedder
from app.retrieval.query_processor import extract_scope
from app.retrieval.retriever import retrieve, validate_scope
from app.retrieval.sparse import _tokenize as base_tokenize
from app.storage.qdrant_store import get_qdrant_client, search_dense
from run_retrieval_eval import has_evidence, is_relevant

ROOT = Path(__file__).resolve().parent.parent
STOPWORDS = frozenset(
    "a an the of to in on at by for with and or is are was were be been being do does did can could may might must shall "
    "should will would what which who whom whose when where how why that this these those it its as from than then there "
    "their they them you your i we our any each per if not no into over under about".split()
)


def stem(token: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenizer_variant(name: str):
    def base(text: str) -> list[str]:
        return base_tokenize(text)

    def nostop(text: str) -> list[str]:
        return [t for t in base_tokenize(text) if t not in STOPWORDS]

    def nostop_stem(text: str) -> list[str]:
        return [stem(t) for t in base_tokenize(text) if t not in STOPWORDS]

    def stem_only(text: str) -> list[str]:
        return [stem(t) for t in base_tokenize(text)]

    return {"base": base, "nostop": nostop, "nostop_stem": nostop_stem, "stem": stem_only}[name]


class VariantBM25:
    def __init__(self, payloads, ids, tokenizer, k1=1.5, b=0.75):
        self.payloads = payloads
        self.ids = ids
        self.tokenizer = tokenizer
        self.bm25 = BM25Okapi([tokenizer(p.get("text", "")) for p in payloads], k1=k1, b=b)

    def search(self, query, top_k, section_filter=None, effective_year_filter=None):
        scores = self.bm25.get_scores(self.tokenizer(query))
        rows = []
        for i, s in enumerate(scores):
            if s <= 0:
                continue
            p = self.payloads[i]
            if section_filter is not None and p.get("section_code") != section_filter:
                continue
            if effective_year_filter is not None and p.get("effective_year") != effective_year_filter:
                continue
            rows.append((s, i))
        rows.sort(key=lambda x: -x[0])
        return [{"score": float(s), "payload": self.payloads[i], "id": self.ids[i]} for s, i in rows[:top_k]]


def weighted_rrf(dense, sparse, top_k, k=60, w_dense=1.0, w_sparse=1.0):
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for rank, r in enumerate(dense, start=1):
        scores[str(r["id"])] = scores.get(str(r["id"]), 0.0) + w_dense / (k + rank)
        payloads[str(r["id"])] = r["payload"]
    for rank, r in enumerate(sparse, start=1):
        scores[str(r["id"])] = scores.get(str(r["id"]), 0.0) + w_sparse / (k + rank)
        payloads[str(r["id"])] = r["payload"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [{"id": i, "score": s, "payload": payloads[i]} for i, s in ranked[:top_k]]


def first_rank(flags):
    for i, f in enumerate(flags, start=1):
        if f:
            return i
    return None


def metrics(rank_pairs):
    n = len(rank_pairs)
    out = {"n": n}
    for k in (1, 3, 5):
        out[f"art@{k}"] = sum(1 for a, _ in rank_pairs if a is not None and a <= k) / n
    for k in (1, 5):
        out[f"ev@{k}"] = sum(1 for _, e in rank_pairs if e is not None and e <= k) / n
    out["mrr"] = sum((1 / a) for a, _ in rank_pairs if a is not None and a <= 10) / n
    return out


class Workbench:
    def __init__(self, dataset="retrieval_eval_v1.jsonl", pool=50):
        self.questions = [json.loads(l) for l in (ROOT / "eval" / dataset).read_text().splitlines() if l.strip()]
        self.client = get_qdrant_client()
        retriever_module.get_qdrant_client = lambda: self.client
        self.embedder = Embedder()
        self.embedder.encode_query("warm up")
        pts = []
        offset = None
        while True:
            recs, offset = self.client.scroll(settings.qdrant_collection, limit=500, offset=offset, with_payload=True)
            pts += recs
            if offset is None:
                break
        self.payloads = [p.payload for p in pts]
        self.ids = [str(p.id) for p in pts]
        self.dense = {}
        self.exact = {}
        self.scopes = {}
        for q in self.questions:
            scope = validate_scope(extract_scope(q["question"]), self.client)
            self.scopes[q["id"]] = scope
            vec = self.embedder.encode_query(q["question"])
            self.dense[q["id"]] = search_dense(
                self.client, settings.qdrant_collection, vec, pool,
                section_filter=scope.section_code, effective_year_filter=scope.season,
            )
            if scope.exact_article:
                res = retrieve(q["question"], bm25_index=None, enable_reranker=False)
                if res.was_exact_lookup:
                    self.exact[q["id"]] = res.candidates

    def bm25(self, tokenizer="base", k1=1.5, b=0.75):
        return VariantBM25(self.payloads, self.ids, tokenizer_variant(tokenizer), k1, b)

    def sparse_lists(self, index, pool=50):
        return {
            q["id"]: index.search(
                q["question"], pool,
                section_filter=self.scopes[q["id"]].section_code,
                effective_year_filter=self.scopes[q["id"]].season,
            )
            for q in self.questions
        }

    def score(self, lists, only=None, depth=5):
        pairs = []
        for q in self.questions:
            if not q["answerable"] or (only and q["id"] not in only):
                continue
            cands = lists[q["id"]][:depth]
            art = first_rank([is_relevant(c["payload"], q) for c in cands])
            ev = first_rank([has_evidence(c["payload"], q) for c in cands])
            pairs.append((art, ev))
        return metrics(pairs)

    def fused(self, sparse, k=60, w_dense=1.0, w_sparse=1.0, pool_d=20, pool_s=20, top=5, use_exact=True):
        out = {}
        for q in self.questions:
            qid = q["id"]
            if use_exact and qid in self.exact:
                out[qid] = self.exact[qid]
            else:
                out[qid] = weighted_rrf(self.dense[qid][:pool_d], sparse[qid][:pool_s], top, k, w_dense, w_sparse)
        return out

    def non_exact_ids(self):
        return {q["id"] for q in self.questions if q["answerable"] and q["id"] not in self.exact}
