from __future__ import annotations

SYSTEM_PROMPT = """\
Answer FIA Formula 1 regulation questions using ONLY the numbered evidence. Evidence is the current 2026 regulation unless marked.
Glossary: Q1/Q2/Q3 = first/second/third qualifying parts; SQ1-SQ3 = sprint qualifying parts; PU = power unit; ICE = engine; ERS = energy recovery; ECU = electronic control unit; RWTT = restricted wind tunnel testing.
Rules:
1. Use only facts in the evidence; no outside knowledge.
2. "asks": the specific information requested (number, limit, name, condition, date).
3. "claims": statements that directly give that information, each with supporting evidence numbers in "sources". Do not add related rules about another aspect of the topic. A related limit for a different purpose is not an answer.
4. Copy numbers, units and symbols exactly. Never convert, round, sum or compute.
5. Cite article numbers only if they appear in the evidence.
6. NON-BINDING COMMENT evidence is not a requirement: if asked what a note says, report it and say it is non-binding. If a provision was replaced, moved or deleted, report that.
7. FUTURE CHANGE evidence is not in force: use it only for questions about that year. Do not write status sentences; the system labels them.
8. If the question may mean different regulations (e.g. teams vs power unit manufacturers), answer each separately.
9. For multi-part questions answer each part separately and name every unanswered part in "not_covered". If the evidence does not state the specific information asked, return no claims and describe the missing information in "not_covered".
10. For a broad question about limits, give the stated numeric limits first.
11. At most 3 claims, one sentence of at most 30 words each.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "asks": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "sources"],
            },
        },
        "not_covered": {"type": "string"},
    },
    "required": ["asks", "claims", "not_covered"],
}


def evidence_label(payload: dict, index: int) -> str:
    parts = [str(payload.get("section_code", "?")) + f" Issue {payload.get('issue', '?')}"]
    article = payload.get("article_id")
    if article:
        title = payload.get("article_title")
        parts.append(f"Art {article}" + (f' "{title}"' if title else ""))
    elif payload.get("appendix_id"):
        parts.append(f"Appendix {payload['appendix_id']}")
    page_start, page_end = payload.get("page_start"), payload.get("page_end")
    if page_start:
        parts.append(f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}")
    if payload.get("is_current") is False:
        year = payload.get("effective_year")
        parts.append(f"FUTURE CHANGE effective {year}, NOT in force" if year else "FUTURE CHANGE, NOT in force")
    if payload.get("is_binding") is False:
        parts.append("NON-BINDING COMMENT")
    return f"[C{index}] " + " | ".join(parts)


def build_context_block(candidates: list[dict]) -> str:
    blocks = []
    for i, cand in enumerate(candidates, start=1):
        p = cand["payload"]
        blocks.append(f"{evidence_label(p, i)}\n{p.get('text', '')}")
    return "\n\n".join(blocks)


def build_user_message(question: str, candidates: list[dict], notes: list[str]) -> str:
    note_block = "\n".join(f"- {n}" for n in notes) if notes else "- none"
    return (
        f"Question: {question}\nNotes:\n{note_block}\n\nEvidence (cite by number):\n\n"
        f"{build_context_block(candidates)}"
    )
