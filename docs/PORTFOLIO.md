# F1 Regulations AI: project description

## Problem

The FIA publishes the Formula 1 regulations as six long PDFs (Sections A–F) that are revised several times a season. Finding a specific rule means knowing which section and article to open, and the documents mix current rules, approved changes for later seasons, non-binding comments and struck-out text. Generic chatbots answer from memory and cannot show where a claim comes from.

## Solution

F1 Regulations AI answers questions in plain English using only the official 2026 FIA documents. Each answer is made of short claims, every claim is checked against the regulation text it cites, and each citation links to the exact page of the official FIA PDF. When the regulations do not support an answer, the system says so and states what is missing instead of guessing.

## Technical architecture

- **Ingestion:** scrapes the FIA catalog, downloads the latest issue of each section, parses PDFs with PyMuPDF (article hierarchy, tables, struck-out provisions) and chunks at Article/Clause boundaries.
- **Index:** 2,805 chunks embedded with all-MiniLM-L6-v2 into an embedded Qdrant store; a SQLite registry tracks document versions, hashes, ingestion runs, deleted provisions and an update journal. Every chunk carries season, section, issue, article path, effective year, in-force and binding flags.
- **Retrieval:** exact article lookup, otherwise dense search plus BM25 fused with Reciprocal Rank Fusion, year-aware scoping, multi-part question splitting and sibling-chunk completion for long clauses.
- **Generation:** a local Ollama model (qwen3.5:9b) returns JSON claims. A verifier checks numbers, units, ordinals and wording against the cited chunk. Citations are built by the backend from chunk metadata.
- **Application:** FastAPI backend and a Streamlit interface that labels non-binding comments, approved future changes, partial answers and truncated context.
- **Operations:** a safe update pipeline with a launchd schedule template (not enabled by default).
- No cloud APIs, agent frameworks or GraphRAG; retrieval and generation are explicit Python.

## Measured performance

All figures come from this repository's evaluation files and tests.

- **Retrieval (54 answerable questions):** article Recall@5 of 0.685 for dense, 0.759 for BM25, 0.778 for hybrid and 0.870 for the full pipeline. On a separate 25-question holdout the pipeline reaches 0.840 (evidence Recall@5 0.864). Median hybrid retrieval takes about 28 ms.
- **Answers (33 + 13 questions, one run, before the final targeted fixes):** all 25 answerable main-set questions were answered; 18/25 passed the strict automatic scorer (7/10 on the fresh set), and manual review showed several misses were valid answers citing neighbouring articles. Citation precision was 0.90 and page accuracy 0.86. 4 of 5 unanswerable questions were refused; the fifth was fixed afterwards and verified on that question only.
- **Latency:** local generation takes about 15–40 s per answer on a 16 GB laptop.
- **Quality gates:** 270 automated tests, including update and recovery scenarios.

## Reliability features

- Backend-resolved citations with page-level links to the official PDFs.
- Per-claim verification; unsupported or computed claims are dropped.
- Explicit abstention, "not covered" reporting and context-truncation warnings.
- Separate handling of current rules, approved future changes and non-binding comments.
- Updates that never leave the index half-changed: staged processing, count and size sanity checks, journaled commits and automatic recovery.
- Single-process storage handled with short lock windows, request retry and a clear 503 during updates.
- Answers keep working when the FIA site is down, because queries never contact it.

## Honest limitations

Generation is slow without a GPU, table-heavy and ambiguous questions can produce partial answers or abstentions, evaluation sets are small, and the update system has been tested with mocks and temporary data but not yet against a real new FIA issue.
