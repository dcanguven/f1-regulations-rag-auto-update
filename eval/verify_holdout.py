from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_ground_truth import ROOT, derive_article, load_pages, norm  # noqa: E402

DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "holdout_en_v1.jsonl"


def related(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def main() -> int:
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    cache: dict[str, list[str]] = {}
    failures = 0
    for r in rows:
        problems = []
        if not r["answerable"]:
            blob = ""
            for sec, issue in (("A", 3), ("B", 9), ("C", 21), ("D", 8), ("E", 7), ("F", 11)):
                doc = f"fia_f1_2026_{sec}_issue_{issue:02d}"
                cache.setdefault(doc, load_pages(doc))
                blob += " ".join(cache[doc])
            hits = {t: len(re.findall(t, blob, re.IGNORECASE)) for t in r["absence_check"]}
            ok = all(v == 0 for v in hits.values())
            print(f"{r['id']:4} {'OK  ' if ok else 'FAIL'} absence {hits}")
            failures += not ok
            continue
        for gi, group in enumerate(r["groups"]):
            for cand in group["candidates"]:
                pages = cache.setdefault(cand["document_id"], load_pages(cand["document_id"]))
                derived = set()
                for ev in cand["evidence"]:
                    if norm(ev["text"]) not in norm(pages[ev["page"] - 1]):
                        problems.append(f"g{gi} evidence not on p{ev['page']}: {ev['text']!r}")
                        continue
                    derived.add(derive_article(pages, ev["page"], ev["text"]))
                if cand["evidence"]:
                    ok = any(d and any(related(d, a) for a in cand["relevant_article_ids"]) for d in derived)
                    if not ok:
                        problems.append(f"g{gi} scan {sorted(x for x in derived if x)} vs {cand['relevant_article_ids']}")
                else:
                    ids_in_doc = set(re.findall(r"(?m)^([A-F]\d{1,3}(?:\.\d{1,3})+)\s*$", "\n".join(pages)))
                    for a in cand["relevant_article_ids"]:
                        if a not in ids_in_doc:
                            problems.append(f"g{gi} article {a} not found as a heading line in the PDF")
        print(f"{r['id']:4} {'OK  ' if not problems else 'FAIL'} {r['category']:17} {' ; '.join(problems)}")
        failures += bool(problems)
    print(f"\n{len(rows)} questions, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
