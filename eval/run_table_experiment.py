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

from qdrant_client import QdrantClient

from app.chunking.chunker import chunk_document
from app.config import settings
from app.ingestion.catalog import SECTION_NAMES, make_document_id
from app.parsing.pdf_parser import parse_pdf
from app.retrieval import retriever as retriever_module
from app.retrieval.embedder import Embedder
from app.retrieval.retriever import retrieve
from app.retrieval.sparse import BM25Index
from app.storage.qdrant_store import chunk_to_point, ensure_collection
from experiment_lib import first_rank, metrics
from run_retrieval_eval import has_evidence, is_relevant

ROOT = Path(__file__).resolve().parent.parent
DOCS = (("A", 3), ("B", 9), ("C", 21), ("D", 8), ("E", 7), ("F", 11))


def main(dataset: str = "retrieval_eval_v1.jsonl", modes=("packed", "row_small", "row")) -> None:
    questions = [json.loads(l) for l in (ROOT / "eval" / dataset).read_text().splitlines() if l.strip()]
    parsed = {}
    for sec, issue in DOCS:
        parsed[sec] = parse_pdf(
            ROOT / "data" / "raw" / "2026" / sec / f"issue_{issue:02d}.pdf",
            make_document_id(2026, sec, issue), 2026, sec, SECTION_NAMES[sec], issue, "https://www.fia.com/x",
        )
    embedder = Embedder()
    embedder.encode_query("warm")
    cache: dict[str, list[float]] = {}
    summary = {}
    for mode in modes:
        settings.table_chunk_mode = mode
        chunks = [c for sec, _ in DOCS for c in chunk_document(parsed[sec])]
        texts = [c.embedding_text for c in chunks if c.embedding_text not in cache]
        t0 = time.perf_counter()
        if texts:
            for text, vec in zip(texts, embedder.encode(texts)):
                cache[text] = vec
        embed_s = time.perf_counter() - t0
        client = QdrantClient(":memory:")
        name = f"exp_{mode}"
        ensure_collection(client, name, settings.embedding_dim)
        points = [chunk_to_point(c, cache[c.embedding_text]) for c in chunks]
        client.upsert(name, points=points)
        bm25 = BM25Index()
        bm25.build([p.payload for p in points], [str(p.id) for p in points])
        settings.qdrant_collection = name
        retriever_module.get_qdrant_client = lambda client=client: client
        pairs = {}
        lat = []
        for q in questions:
            t = time.perf_counter()
            res = retrieve(q["question"], bm25_index=bm25, enable_reranker=False)
            lat.append((time.perf_counter() - t) * 1000)
            if q["answerable"]:
                cands = res.candidates[:10]
                pairs[q["id"]] = (
                    first_rank([is_relevant(c["payload"], q) for c in cands]),
                    first_rank([has_evidence(c["payload"], q) for c in cands]),
                    q["category"],
                )
        cats = defaultdict(list)
        for a, e, c in pairs.values():
            cats[c].append((a, e))
        summary[mode] = {
            "chunks": len(chunks),
            "table_chunks": sum(c.contains_table for c in chunks),
            "embed_seconds_new": round(embed_s, 1),
            "overall": metrics([(a, e) for a, e, _ in pairs.values()]),
            "by_category": {c: metrics(v) for c, v in cats.items()},
            "latency_median_ms": round(statistics.median(lat), 1),
            "latency_p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
            "per_question": {qid: [a, e] for qid, (a, e, _) in pairs.items()},
        }
        client.close()
    out = ROOT / "eval" / "results" / "table_experiment.json"
    out.write_text(json.dumps(summary, indent=1))
    for mode, s in summary.items():
        m = s["overall"]
        print(f"{mode:10} chunks={s['chunks']} table_chunks={s['table_chunks']} | art R@1={m['art@1']:.3f} R@3={m['art@3']:.3f} R@5={m['art@5']:.3f} | ev R@1={m['ev@1']:.3f} R@5={m['ev@5']:.3f} | MRR={m['mrr']:.3f} | lat {s['latency_median_ms']}/{s['latency_p95_ms']}ms")
        for c in ("numeric_table", "aero_testing_F", "financial", "exact_article"):
            mc = s["by_category"][c]
            print(f"     {c:15} art R@1={mc['art@1']:.2f} R@5={mc['art@5']:.2f} ev R@1={mc['ev@1']:.2f} R@5={mc['ev@5']:.2f}")


if __name__ == "__main__":
    main()
