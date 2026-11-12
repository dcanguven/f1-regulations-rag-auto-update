from __future__ import annotations

import re
import unicodedata

FF_LIGATURE_REPAIRS: dict[str, str] = {
    "oa": "off",
    "eaect": "effect",
    "eaects": "effects",
    "eaort": "effort",
    "eaorts": "efforts",
    "eaiciency": "efficiency",
    "eaicient": "efficient",
    "diaer": "differ",
    "diaerent": "different",
    "diaerence": "difference",
    "diaerential": "differential",
    "diaerentiate": "differentiate",
    "stianess": "stiffness",
    "stiaening": "stiffening",
    "aaect": "affect",
    "aaected": "affected",
    "aaecting": "affecting",
    "aaix": "affix",
    "suaix": "suffix",
    "suaicient": "sufficient",
    "suaiciently": "sufficiently",
    "paraainic": "paraffinic",
    "oaicially": "officially",
    "oaset": "offset",
    "oaer": "offer",
    "buaer": "buffer",
    "midria": "midriff",
    "diaerences": "differences",
    "diaerentiates": "differentiates",
    "diaering": "differing",
    "diaers": "differs",
    "oaers": "offers",
    "paraains": "paraffins",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_PROTECTED_DECOMPOSITIONS = ("<super>", "<sub>", "<fraction>")


def _is_protected(ch: str) -> bool:
    if ch == "˚":
        return True
    return unicodedata.decomposition(ch).startswith(_PROTECTED_DECOMPOSITIONS)


def normalize_unicode(text: str) -> str:
    out: list[str] = []
    run: list[str] = []
    for ch in text:
        if ord(ch) > 127 and _is_protected(ch):
            if run:
                out.append(unicodedata.normalize("NFKC", "".join(run)))
                run = []
            out.append(ch)
        else:
            run.append(ch)
    if run:
        out.append(unicodedata.normalize("NFKC", "".join(run)))
    return "".join(out)


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _repair_token(match: re.Match) -> str:
    token = match.group(0)
    if token.isupper() and len(token) <= 3:
        return token
    fixed = FF_LIGATURE_REPAIRS.get(token.lower())
    if fixed is None:
        return token
    return _match_case(token, fixed)


def repair_corrupted_words(text: str) -> str:
    return _TOKEN_RE.sub(_repair_token, text)


def clean_text(text: str) -> str:
    return repair_corrupted_words(normalize_unicode(text))
