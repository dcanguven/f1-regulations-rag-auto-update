from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pymupdf

from app.parsing.text_repair import clean_text

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "retrieval_eval_v1.jsonl"
ID_LINE = re.compile(r"(?m)^([A-F]\d{1,3}(?:\.\d{1,3})+)(?:[ \t]+[A-Z][^\n]{3,80})?[ \t]*$")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def load_pages(document_id: str) -> list[str]:
    sec = document_id.split("_")[3]
    issue = int(document_id.rsplit("_", 1)[1])
    pdf = pymupdf.open(ROOT / "data" / "raw" / "2026" / sec / f"issue_{issue:02d}.pdf")
    return [clean_text(p.get_text()) for p in pdf]


def derive_article(pages: list[str], page: int, evidence: str) -> str | None:
    target = norm(evidence)
    page_text = pages[page - 1]
    flat_page = norm(page_text)
    if target not in flat_page:
        return None
    pos_flat = flat_page.index(target)
    pos = None
    count = 0
    for m in re.finditer(r"\S+", page_text):
        if count == len(flat_page[:pos_flat].split()):
            pos = m.start()
            break
        count += 1
    if pos is None:
        pos = 0
    before = "\n".join(pages[: page - 1]) + "\n" + page_text[:pos]
    ids = ID_LINE.findall(before)
    return ids[-1] if ids else None


def main() -> int:
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    cache: dict[str, list[str]] = {}
    failures = 0
    for r in rows:
        if not r["answerable"]:
            terms = r.get("absence_check")
            if terms:
                texts = []
                for sec, issue in (("A", 3), ("B", 9), ("C", 21), ("D", 8), ("E", 7), ("F", 11)):
                    doc = f"fia_f1_2026_{sec}_issue_{issue:02d}"
                    cache.setdefault(doc, load_pages(doc))
                    texts.append(" ".join(cache[doc]))
                blob = " ".join(texts)
                hits = {t: len(re.findall(t, blob, re.IGNORECASE)) for t in terms}
                ok = all(v == 0 for v in hits.values())
                print(f"{r['id']:6} {'OK  ' if ok else 'FAIL'} absence {hits}")
                failures += not ok
            else:
                print(f"{r['id']:6} OK   no ground truth (scope-behaviour question)")
            continue
        pages = cache.setdefault(r["document_id"], load_pages(r["document_id"]))
        problems = []
        derived = set()
        for ev in r["evidence"]:
            page_text = norm(pages[ev["page"] - 1])
            if norm(ev["text"]) not in page_text:
                problems.append(f"evidence not on p{ev['page']}: {ev['text']!r}")
                continue
            d = derive_article(pages, ev["page"], ev["text"])
            derived.add(d)
        for ev in r.get("forbidden_evidence", []):
            if r["effective_year"] == 2026 and not any(ev in norm(p) for p in pages):
                problems.append(f"forbidden evidence not present anywhere: {ev!r}")
        rel = r["relevant_article_ids"]
        id_ok = any(d and any(d == a or d.startswith(a + ".") or a.startswith(d + ".") for a in rel) for d in derived)
        if not id_ok:
            problems.append(f"independent article scan {sorted(x for x in derived if x)} vs labelled {rel}")
        print(f"{r['id']:6} {'OK  ' if not problems else 'FAIL'} p{r['evidence'][0]['page']} labelled={rel} scanned={sorted(x for x in derived if x)} {' ; '.join(problems)}")
        failures += bool(problems)
    print(f"\n{len(rows)} questions, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
