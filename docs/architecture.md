# Architecture

## Overview

A local-first RAG system for the official FIA Formula 1 2026 regulations. Ingestion and querying are separate paths: only `python -m app.ingestion.sync` contacts the FIA, and `/ask` reads Qdrant and SQLite only.

## Components

```
INGESTION / UPDATE   (python -m app.ingestion.sync)
  catalog.py        scrape https://www.fia.com/regulation/category/110, pick the latest issue per section
  updater.py        plan, process (no Qdrant), commit (short, lock held), recover
  downloader.py     httpx download to .part, SHA-256
  pdf_parser.py     PyMuPDF, hierarchy tracking, struck-text detection
  chunker.py        Article/Clause-boundary chunks, deleted-provision extraction
  embedder.py       all-MiniLM-L6-v2, 384-d, CPU
  qdrant_store.py   embedded Qdrant (data/qdrant), collection f1_regulations_minilm
  sqlite_store.py   documents, ingestion_log, deleted_provisions, update_journal

QUERY   (Streamlit -> FastAPI POST /ask)
  query_processor.py  article reference + scope + year classification
  retriever.py        exact article lookup | dense + BM25 + RRF, year scope, sibling completion
  prompt.py           glossary, evidence labels, JSON schema {asks, claims, not_covered}
  llm.py              Ollama (qwen3.5:9b), think off, JSON-schema output
  answer.py           claim filtering, quantity gate, missing-information handling
  verification.py     number / unit / ordinal / lexical support checks per claim
  citation.py         citations resolved from candidate metadata
```

## Query flow

1. The question is normalised and scoped (season, section, article reference, year mention).
2. An explicit article reference bypasses vector search and scrolls that article. Otherwise dense search (Qdrant) and BM25 (in memory) are fused with RRF (k=60). Multi-part questions are split and retrieved per part.
3. Year handling: current 2026 text is the default. A question about 2027–2030 uses approved future provisions, which are labelled "not currently in force". A year that only appears as a table column in current text is answered from the current rules.
4. The top chunks plus completion siblings (the best other chunk of the top articles, the lead-in of long list articles, siblings of heading-only chunks) are given to the model as numbered evidence.
5. The model returns claims with source numbers. Each claim is verified against its cited chunk; unsupported claims are dropped or re-attributed. Status-only and absence statements are removed. If the model reports that the requested information is missing, or a quantity question has no verified number, the system abstains and says what is not covered.
6. Citations (section, article, issue, pages, FIA PDF URL, binding and in-force flags) are built from chunk metadata, never from model text.

## Version-aware data model

Each chunk carries `season`, `section_code`, `issue`, `article_id`/`article_path`, `effective_year`, `is_current`, `is_binding`, page range, `source_url` and `pipeline_version`. The live index holds one issue per section (A3, B9, C21, D8, E7, F11; 2,805 vectors): 2,686 current 2026 chunks and 119 future-effective chunks (2027–2030). Deleted provisions (87 rows) live in SQLite and are never indexed as current text.

## Update and concurrency design

Embedded Qdrant is single-process. The API opens it briefly per request, so the update system keeps its Qdrant window short:

1. Check the catalog (no Qdrant). If nothing is newer, only an `ingestion_log` row is written.
2. For changed sections, download to `.part`, verify it is a PDF, parse, chunk and embed (no Qdrant).
3. Open Qdrant (retrying while the API holds it), recover any interrupted journal entries, then per section: journal start, upsert, exact count check, chunk-count sanity check against the indexed issue, SQLite transaction that switches the latest issue and deleted provisions, delete superseded vectors, journal commit.
4. Bump `data/.index_generation`. The API compares it on each request and rebuilds BM25 and the article cache.

Interrupted before activation: partial points of the new issue are removed and the old issue stays. Interrupted after activation: the next run deletes the superseded vectors. An interrupted in-place (same-issue) update is retried on the next run. A file lock prevents two updates at once. API requests that cannot get Qdrant within `QDRANT_LOCK_WAIT_SECONDS` return HTTP 503.

## Design decisions

- **Structure-aware chunking:** articles are never merged; list articles keep clause boundaries.
- **Hybrid retrieval with exact lookup:** explicit article references are deterministic.
- **Backend-resolved citations:** the model cannot invent metadata.
- **Abstention over guessing:** unsupported, unverifiable or missing-information cases return a clear explanation.
- **No agents or graph frameworks:** retrieval and generation are explicit Python; LangChain is used only to call Ollama.
- **Evaluate before adding complexity:** the reranker stays disabled; context expansion is off.

## Known limitations

See the README. Main points: local generation latency, table and list evidence gaps, partial answers to ambiguous questions, single-process embedded Qdrant, English-only evaluation.
