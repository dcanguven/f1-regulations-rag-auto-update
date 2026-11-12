from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

_ID_RE = re.compile(r"\b[A-F]\d{1,3}(?:\.\d{1,3})+(?:\.[a-z])?(?![\w])")
_NUM_RE = re.compile(r"\d+(?:['.,]\d+)*")
_UNIT_RE = re.compile(
    r"(?<![a-z\d.,'])(\d+(?:['.,]\d+)*)\s?(%|°|km/h|kg|kn|kw|mm|cm|m/s|mj/h|mj|ms|barg|bar|seconds?|minutes?|hours?|days?|laps?)(?![a-z])",
    re.IGNORECASE,
)
_ATTACHED_UNIT_RE = re.compile(r"(?<![a-z\d.,'])(\d+(?:['.,]\d+)*)([a-z%°/]{1,8})(?![a-z])", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b")
_ORDINALS = frozenset({"st", "nd", "rd", "th", "s"})
_STOPWORDS = frozenset(
    "this that these those there their with from have has been which when where into than then also only each such other "
    "while about must shall will would could should does done being were they them what whose whom than article section "
    "regulation regulations current future approved change changes effective binding comment states stated says according "
    "provided evidence applies apply under over after before between within without during".split()
)


_TIME_RE = re.compile(r"(?<![\d.:])(\d{1,2}):(\d{2})(?![\d:])")
_GROUPED_RE = re.compile(r"(?<![\d.,'])(\d{1,3})((?:[,'](?=\d{3}(?!\d))\d{3})+)(?!\d)")


def canonical_numbers(text: str) -> str:
    text = _TIME_RE.sub(r"\1.\2", text)
    return _GROUPED_RE.sub(lambda m: m.group(1) + re.sub(r"[,']", "", m.group(2)), text)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)


@dataclass
class ClaimVerdict:
    text: str
    sources: list[int]
    valid: bool
    problems: list[str] = field(default_factory=list)
    support: float = 1.0


def _prefixes(text: str) -> set[str]:
    return {t[:5] for t in re.findall(r"[a-z]{4,}", text)}


def _number_present(token: str, source: str) -> bool:
    pattern = r"(?<!\d)(?<!\d[.,'])" + re.escape(token) + r"(?!\d|[.,']\d)"
    return re.search(pattern, source) is not None


def verify_claim(
    text: str,
    sources: list[int],
    evidence: dict[int, dict[str, Any]],
    question: str,
) -> ClaimVerdict:
    problems: list[str] = []
    valid_sources = [s for s in dict.fromkeys(sources) if s in evidence]
    if not valid_sources:
        problems.append("invalid_source_reference")
        return ClaimVerdict(text, [], False, problems + ["no_valid_source"], 0.0)

    payloads = [evidence[s] for s in valid_sources]
    source_text = canonical_numbers(normalise(" ".join(p.get("text", "") for p in payloads)))
    extra_numbers = {str(p.get(k)) for p in payloads for k in ("effective_year", "issue", "page_start", "page_end") if p.get(k)}
    question_norm = canonical_numbers(normalise(question))

    claim_norm = canonical_numbers(normalise(text))
    allowed_ids = set()
    for p in payloads:
        aid = p.get("article_id")
        if aid:
            parts = aid.split(".")
            allowed_ids.update(".".join(parts[: i + 1]).lower() for i in range(len(parts)))
        allowed_ids.update(i.lower() for i in _ID_RE.findall(p.get("text", "")))
    claim_ids = [i.lower() for i in _ID_RE.findall(text)]
    for cid in claim_ids:
        if cid not in allowed_ids and cid not in question_norm:
            problems.append(f"unsupported_article_reference:{cid}")
    stripped = _ID_RE.sub(" ", text)
    stripped_norm = canonical_numbers(normalise(stripped))

    for token in dict.fromkeys(_NUM_RE.findall(stripped_norm)):
        if token in extra_numbers or _number_present(token, question_norm):
            continue
        if not _number_present(token, source_text):
            problems.append(f"unsupported_number:{token}")
    for number, unit in _UNIT_RE.findall(stripped_norm):
        unit_stem = unit.rstrip("s") if unit.isalpha() else unit
        if not re.search(re.escape(number) + r"\s?" + re.escape(unit_stem), source_text):
            if number not in extra_numbers and not _number_present(number, question_norm):
                problems.append(f"unsupported_unit:{number} {unit}")

    for number, suffix in _ATTACHED_UNIT_RE.findall(stripped_norm):
        if suffix in _ORDINALS or any(f"{number} {suffix}" in p or f"{number}{suffix}" in p for p in problems):
            continue
        if not re.search(re.escape(number) + r"\s?" + re.escape(suffix), source_text):
            if number not in extra_numbers and not _number_present(number, question_norm):
                problems.append(f"unsupported_unit:{number}{suffix}")

    for ordinal in dict.fromkeys(_ORDINAL_RE.findall(stripped_norm)):
        if ordinal not in source_text:
            problems.append(f"unsupported_ordinal:{ordinal}")

    claim_tokens = [t for t in re.findall(r"[a-z]{4,}", claim_norm) if t not in _STOPWORDS]
    support = 1.0
    if len(claim_tokens) >= 3:
        source_prefixes = _prefixes(source_text) | _prefixes(question_norm)
        support = sum(1 for t in claim_tokens if t[:5] in source_prefixes) / len(claim_tokens)
        anchored = not problems and bool(_NUM_RE.findall(stripped_norm) or _ORDINAL_RE.findall(stripped_norm))
        threshold = settings.claim_anchored_min_support if anchored else settings.claim_min_support
        if support < threshold:
            problems.append(f"low_lexical_support:{support:.2f}")
    return ClaimVerdict(text, valid_sources, not problems, problems, support)
