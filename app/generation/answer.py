from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from app.config import settings
from app.generation.citation import resolve_citations
from app.generation.llm import INSUFFICIENT_EVIDENCE, generate_structured
from app.generation.prompt import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_message
from app.generation.verification import verify_claim
from app.models.schemas import AnswerClaim, AskResponse, QueryScope
from app.retrieval.query_processor import split_question

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 14
MAX_CLAIMS = 8
PROMPT_OVERHEAD_TOKENS = 700
_STATUS_ONLY_RE = re.compile(
    r"^(?:non-binding comment:\s*)?(?:this|the|these|article\s+\S+)\b.{0,80}?\b"
    r"(?:approved future change|not (?:currently )?in force|(?:is|are) (?:a )?non-binding (?:comment|note)s?|not a regulatory requirement)\b",
    re.IGNORECASE,
)
_ABSENCE_RE = re.compile(
    r"\b(?:do(?:es)? not|doesn't|don't|did not|cannot|can't|no)\s+(?:state|specify|mention|define|contain|provide|include|say|give|list|cover|address)\b"
    r"|\bnot (?:stated|specified|mentioned|defined|provided|covered|included)\b|\bevidence (?:does not|doesn't|provided)\b|\bno (?:specific|explicit|stated)\b",
    re.IGNORECASE,
)
_QUANTITY_Q_RE = re.compile(
    r"\bhow (?:many|much|long|far|fast|strong|wide|high|heavy|often|big|large)\b"
    r"|\bwhat (?:is|are) the (?:\w+\s+){0,2}(?:maximum|minimum|limit|limits|number|amount|speed|distance|mass|weight|width|diameter|size|fine|duration|length|period|allowance)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be been can could may might must should shall will would do does did "
    "what which who whom whose when where why how many much any all each every there their its it this that these those with by from "
    "as if than then also not no about into over under left out kind type regulations regulation rule rules formula one allowed".split()
)
_NUMBER_WORDS = frozenset(
    "single one two three four five six seven eight nine ten eleven twelve both no none zero unlimited any".split()
)


def _focus_terms(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _overlap(terms: set[str], other: set[str]) -> float:
    return len(terms & other) / len(terms) if terms else 0.0


def _year_notes(year_resolution: Any) -> tuple[list[str], Optional[dict]]:
    if year_resolution is None:
        return [], None
    mode, year = year_resolution.mode, year_resolution.year
    info = {"mode": mode, "year": year, "reason": year_resolution.reason, "explicit": year_resolution.explicit}
    if mode == "future" and year_resolution.explicit:
        note = f"The question asks about approved changes effective {year}. The evidence is APPROVED FUTURE CHANGE material; say that it is not currently in force."
    elif mode == "future":
        note = (
            f"The question mentions {year}, but no current regulation text mentions {year}. Any evidence below is an approved future "
            f"provision that is NOT in force today; state that clearly and note that the question is ambiguous about the year."
        )
    elif mode == "current":
        note = f"The question mentions {year}. Current 2026 regulation text lists values for {year}, so answer from the CURRENT evidence."
    elif mode == "ambiguous_both":
        note = (
            f"Both the current rule and an approved change effective {year} exist for the same article. "
            f"State them separately and say which one is in force today."
        )
    elif mode == "past":
        note = f"Only the 2026 regulations are indexed. The question refers to {year}; answer only if the evidence explicitly covers it."
    else:
        note = f"The question mentions {year} but the evidence does not mention it; do not assume a year."
    return [note], info


def _label(sources: list[int], evidence: dict[int, dict[str, Any]]) -> tuple[str, str]:
    payloads = [evidence[s] for s in sources]
    prefix = ""
    suffix = ""
    if all(p.get("is_binding") is False for p in payloads):
        prefix = "Non-binding comment: "
    elif any(p.get("is_binding") is False for p in payloads):
        suffix += " (partly based on a non-binding comment)"
    future = [p for p in payloads if p.get("is_current") is False]
    if future:
        years = sorted({str(p.get("effective_year")) for p in future if p.get("effective_year")})
        when = f" effective {', '.join(years)}" if years else ""
        suffix += f" (approved future change{when}, not currently in force)"
    return prefix, suffix


def _abstain(
    reason: str,
    scope: QueryScope,
    year_info: Optional[dict],
    warnings: list[str],
    debug: Optional[dict],
    extra: str = "",
) -> AskResponse:
    return AskResponse(
        answer=INSUFFICIENT_EVIDENCE + (f" {extra}" if extra else ""),
        scope_used=_scope_dict(scope),
        citations=[],
        debug=debug,
        abstained=True,
        abstain_reason=reason,
        warnings=warnings,
        year_interpretation=year_info,
    )


def generate(
    question: str,
    candidates: list[dict[str, Any]],
    scope: QueryScope,
    debug: bool = False,
    year_resolution: Any = None,
    exact_total_chunks: int = 0,
    exact_truncated: bool = False,
) -> AskResponse:
    t0 = time.perf_counter()
    notes, year_info = _year_notes(year_resolution)
    warnings: list[str] = []
    dbg: Optional[dict] = {} if debug else None

    usable = [c for c in candidates if c["payload"].get("text", "").strip()][:MAX_EVIDENCE]
    if not usable:
        extra = "The regulations indexed cover the 2026 season only." if year_info and year_info["mode"] == "past" else ""
        return _abstain("no_evidence", scope, year_info, warnings, dbg, extra)

    evidence = {i: c["payload"] for i, c in enumerate(usable, start=1)}
    if exact_truncated:
        notes.append(
            f"Only {len(usable)} of {exact_total_chunks} chunks of the requested article were provided, so the evidence may be incomplete; say so."
        )
        warnings.append(f"Evidence truncated: {len(usable)} of {exact_total_chunks} chunks of the requested article were used.")

    parts = split_question(question)
    if len(parts) > 1:
        notes.append("The question has separate parts; answer each part separately: " + " | ".join(f"({i}) {p}" for i, p in enumerate(parts, start=1)))
    user_msg = build_user_message(question, usable, notes)
    budget = settings.llm_num_ctx - settings.llm_max_tokens - PROMPT_OVERHEAD_TOKENS
    while len(user_msg) // 3 > budget and len(usable) > 1:
        usable.pop()
        warnings.append("Evidence was shortened to fit the model context window.")
        user_msg = build_user_message(question, usable, notes)
    evidence = {i: c["payload"] for i, c in enumerate(usable, start=1)}
    llm_meta: dict[str, Any] = {}
    try:
        data = generate_structured(SYSTEM_PROMPT, user_msg, ANSWER_SCHEMA, meta=llm_meta)
    except RuntimeError as e:
        return AskResponse(
            answer=f"Generation failed: {e}",
            scope_used=_scope_dict(scope),
            citations=[],
            abstained=True,
            abstain_reason="generation_failed",
            warnings=warnings,
            year_interpretation=year_info,
            context_truncated=exact_truncated,
        )
    gen_ms = (time.perf_counter() - t0) * 1000

    truncation_note = (
        f"Only {len(usable)} of {exact_total_chunks} chunks of the requested article were considered; ask about a specific clause for a complete answer."
        if exact_truncated else ""
    )
    raw_claims = data.get("claims") or []
    if data.get("can_answer") is False or not raw_claims:
        resp = _abstain("model_declined", scope, year_info, warnings, dbg, truncation_note)
        resp.context_truncated = exact_truncated
        _fill_debug(resp, dbg, debug, gen_ms, usable, data, [], [], llm_meta)
        return resp

    verified: list[AnswerClaim] = []
    rejected: list[AnswerClaim] = []
    seen_text: set[str] = set()
    for raw in raw_claims[:MAX_CLAIMS]:
        text = str(raw.get("text", "")).strip()
        sources = [s for s in raw.get("sources", []) if isinstance(s, int)]
        if not text or text in seen_text or _STATUS_ONLY_RE.match(text):
            continue
        if _ABSENCE_RE.search(text):
            warnings.append("A statement about what the regulations do not say was removed because absence cannot be verified from retrieved evidence.")
            continue
        seen_text.add(text)
        verdict = verify_claim(text, sources, evidence, question)
        if not verdict.valid:
            for alt in evidence:
                if alt in verdict.sources:
                    continue
                retry = verify_claim(text, [alt], evidence, question)
                if retry.valid:
                    warnings.append(f"Citation for a statement was corrected from {['C%d' % s for s in sources]} to C{alt} because only C{alt} supports it.")
                    verdict = retry
                    break
        if verdict.valid and len(verdict.sources) > 1:
            single = [x for x in verdict.sources if verify_claim(text, [x], evidence, question).valid]
            if single:
                verdict.sources = single
        dropped = sorted(set(sources) - set(verdict.sources))
        if verdict.valid and dropped:
            warnings.append(f"Ignored invalid source reference(s) {dropped} returned by the model.")
        claim = AnswerClaim(text=text, sources=[f"C{s}" for s in verdict.sources], verified=verdict.valid, problems=verdict.problems)
        (verified if verdict.valid else rejected).append(claim)

    not_covered = str(data.get("not_covered") or "").strip()
    if not_covered.lower() in ("none", "n/a", "nothing", "-"):
        not_covered = ""
    if verified and len(parts) == 1 and not_covered and _overlap(_focus_terms(question), _focus_terms(not_covered)) >= 0.5:
        warnings.append("The model reported that the specific information asked for is not stated in the evidence.")
        resp = _abstain("requested_information_missing", scope, year_info, warnings, dbg, f"Not covered by the available evidence: {not_covered.rstrip('.')}.")
        resp.claims = verified + rejected
        resp.context_truncated = exact_truncated
        _fill_debug(resp, dbg, debug, gen_ms, usable, data, verified, rejected, llm_meta)
        return resp

    if not verified:
        warnings.append(f"All {len(rejected)} generated claims failed verification against the cited sources.")
        resp = _abstain("no_verified_claims", scope, year_info, warnings, dbg, truncation_note)
        resp.claims = rejected
        resp.context_truncated = exact_truncated
        _fill_debug(resp, dbg, debug, gen_ms, usable, data, verified, rejected, llm_meta)
        return resp

    if len(split_question(question)) == 1 and _QUANTITY_Q_RE.search(question) and not any(_has_quantity(c.text) for c in verified):
        warnings.append("The question asks for a quantity but no verified statement contains one.")
        resp = _abstain("no_quantity_in_evidence", scope, year_info, warnings, dbg, (truncation_note + " " if truncation_note else "") + "Not covered by the available evidence: the specific figure or limit requested.")
        resp.claims = verified + rejected
        resp.context_truncated = exact_truncated
        _fill_debug(resp, dbg, debug, gen_ms, usable, data, verified, rejected, llm_meta)
        return resp

    lines: list[str] = []
    used: list[int] = []
    for claim in verified:
        idx = [int(s[1:]) for s in claim.sources]
        used.extend(idx)
        prefix, suffix = _label(idx, evidence)
        refs = "".join(f"[C{i}]" for i in sorted(set(idx)))
        lines.append(f"{prefix}{claim.text.rstrip()}{suffix} {refs}")
    if rejected:
        warnings.append(f"{len(rejected)} generated statement(s) were removed because they could not be verified against the cited sources.")
    answer = "\n".join(lines)
    missing: list[str] = [not_covered.rstrip(".")] if not_covered and len(not_covered) <= 240 else []
    if len(parts) > 1:
        claim_terms = set().union(*(_focus_terms(c.text) for c in verified))
        for part in parts:
            terms = _focus_terms(part)
            if _overlap(terms, claim_terms) < 0.34 and not any(_overlap(terms, _focus_terms(m)) >= 0.5 for m in missing):
                missing.append(part.rstrip("?. "))
    if missing:
        answer += f"\nNot covered by the available evidence: {'; '.join(missing)}."
        warnings.append("Parts of the question are not covered by the available evidence.")
    if exact_truncated:
        answer += f"\nNote: only {len(usable)} of {exact_total_chunks} chunks of the requested article were considered, so this answer may be incomplete."
    if rejected:
        answer += "\nNote: some generated statements were omitted because they could not be verified against the sources."

    response = AskResponse(
        answer=answer,
        scope_used=_scope_dict(scope),
        citations=resolve_citations(used, evidence),
        claims=verified + rejected,
        warnings=warnings,
        year_interpretation=year_info,
        context_truncated=exact_truncated,
    )
    _fill_debug(response, dbg, debug, gen_ms, usable, data, verified, rejected, llm_meta)
    return response


def _has_quantity(text: str) -> bool:
    return bool(re.search(r"\d", text)) or bool(set(re.findall(r"[a-z]+", text.lower())) & _NUMBER_WORDS)


def _fill_debug(resp, dbg, debug, gen_ms, usable, data, verified, rejected, llm_meta) -> None:
    if not debug:
        return
    resp.debug = {
        "generation_ms": round(gen_ms, 1),
        "num_evidence": len(usable),
        "num_verified_claims": len(verified),
        "num_rejected_claims": len(rejected),
        "rejected": [{"text": c.text, "problems": c.problems} for c in rejected],
        "model_asks": data.get("asks"),
        "model_not_covered": data.get("not_covered"),
        "evidence_articles": [c["payload"].get("article_id") for c in usable],
        "llm": llm_meta,
    }


def _scope_dict(scope: QueryScope) -> dict:
    return {
        "season": scope.season,
        "section_code": scope.section_code,
        "issue": scope.issue,
        "exact_article": scope.exact_article,
        "year_mentioned": scope.year_mentioned,
        "year_mode": scope.year_mode,
    }
