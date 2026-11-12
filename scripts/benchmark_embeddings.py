from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

from app.chunking.chunker import chunk_document
from app.config import settings
from app.ingestion.catalog import SECTION_NAMES, make_document_id
from app.parsing.pdf_parser import parse_pdf
from app.retrieval.embedder import Embedder

SIZES = (100, 500, 1000)
WARMUP = 64


def load_chunks(season: int) -> list:
    chunks = []
    for pdf in sorted((settings.raw_data_dir / str(season)).glob("*/issue_*.pdf")):
        section = pdf.parent.name
        issue = int(re.search(r"issue_(\d+)", pdf.name).group(1))
        parsed = parse_pdf(
            pdf_path=pdf,
            document_id=make_document_id(season, section, issue),
            season=season,
            section_code=section,
            section_name=SECTION_NAMES.get(section, section),
            issue=issue,
            source_url="local",
        )
        chunks.extend(chunk_document(parsed))
    return chunks


def sample(texts: list[str], n: int) -> list[str]:
    if n >= len(texts):
        return [texts[i % len(texts)] for i in range(n)]
    step = len(texts) / n
    return [texts[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--season", type=int, default=settings.default_season)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=300.0)
    args = ap.parse_args()

    t = time.perf_counter()
    chunks = load_chunks(args.season)
    prep = time.perf_counter() - t
    texts = [c.embedding_text for c in chunks]
    total = len(texts)
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]

    if args.device:
        settings.embedding_device = args.device
    if args.batch_size:
        settings.embedding_batch_size = args.batch_size
    if args.model:
        settings.embedding_model = args.model
    if args.max_seq_length is not None:
        settings.embedding_max_seq_length = args.max_seq_length

    embedder = Embedder()
    t = time.perf_counter()
    embedder._load()
    init = time.perf_counter() - t

    tok = embedder._model.tokenizer
    lengths = [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    limit = embedder._model.max_seq_length
    over = sum(1 for n in lengths if n > limit)

    print(f"model={settings.embedding_model} device={settings.embedding_device} "
          f"batch_size={settings.embedding_batch_size} dim={embedder.dim} max_seq_length={limit}")
    print(f"input sha256={digest}")
    print(f"local documents chunked: {total} chunks (parse+chunk {prep:.1f}s); "
          f"tokens max={max(lengths)} mean={sum(lengths)/total:.0f}; over limit: {over}")
    print(f"model init: {init:.2f}s")

    embedder.encode(sample(texts, WARMUP))
    print(f"warm-up done, dim={embedder.dim}")

    print("| Benchmark | Chunks | Time (s) | Chunks/s | Est. full embedding (s) |")
    print("|---|---:|---:|---:|---:|")
    spent = 0.0
    rate = None
    for i, n in enumerate(SIZES, start=1):
        if rate and spent + n / rate > args.time_budget:
            print(f"| {i} | {n} | skipped: projected {n / rate:.0f}s exceeds remaining budget | | |")
            continue
        batch = sample(texts, n)
        t = time.perf_counter()
        vecs = embedder.encode(batch)
        dt = time.perf_counter() - t
        spent += dt
        rate = n / dt
        assert len(vecs) == n and len(vecs[0]) == embedder.dim
        print(f"| {i} | {n} | {dt:.2f} | {rate:.1f} | {total / rate:.1f} ({total / rate / 60:.1f} min) |")


if __name__ == "__main__":
    main()
