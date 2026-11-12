from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pymupdf
from qdrant_client import QdrantClient

from app.config import settings
from app.parsing.text_repair import clean_text

ROOT = Path(__file__).resolve().parent.parent
ID_LINE = re.compile(r"(?m)^([A-F]\d{1,3}(?:\.\d{1,3})+)(?:[ \t]+[A-Z][^\n]{3,80})?[ \t]*$")
CAPTION = re.compile(
    r"(?m)^(?:Changes for (20\d\d)|Approved changes for[^\n]*?(?:starting|commencing|beginning)\s+(?:from\s+)?(?:\d{1,2}\s+\w+\s+)?(20\d\d))",
    re.IGNORECASE,
)
FUTURE_HEAD = re.compile(r"(?m)^APPENDIX [A-F]\d+: [^\n]*SUBSEQUENT YEARS")
FOOTER = re.compile(r"\d{1,2} (?:June|July|August|September|October|November) 2026 ?\s*20\d\d Formula 1")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def related(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def load_docs():
    docs = {}
    for sec in "ABCDEF":
        paths = sorted((ROOT / "data" / "raw" / "2026" / sec).glob("issue_*.pdf"))
        for path in paths:
            doc_id = f"fia_f1_2026_{sec}_issue_{int(path.stem.split('_')[1]):02d}"
            docs[doc_id] = [clean_text(p.get_text()) for p in pymupdf.open(path)]
    return docs


def audit(chunks: list[dict], docs: dict) -> dict:
    res = {"total": len(chunks), "skipped_table_start": 0, "page_ok": 0, "page_off_by_one": 0, "page_not_found": 0,
           "article_checked": 0, "article_ok": 0, "article_mismatch": [], "year_checked": 0, "year_ok": 0,
           "year_mismatch": [], "footer_leak": 0, "page_not_found_examples": []}
    first_body = {}
    for d, pages in docs.items():
        first_body[d] = None
    for ch in chunks:
        pages = docs[ch["document_id"]]
        text = ch["text"]
        if FOOTER.search(text):
            res["footer_leak"] += 1
        if ch["content_type"] == "table":
            res["skipped_table_start"] += 1
            continue
        words = norm(text).split()[:7]
        if len(words) < 4:
            res["skipped_table_start"] += 1
            continue
        pattern = r"\s+".join(re.escape(w) for w in words)
        located = None
        for delta, key in ((0, "page_ok"), (-1, "page_off_by_one"), (1, "page_off_by_one")):
            pn = ch["page_start"] + delta
            if 1 <= pn <= len(pages):
                m = re.search(pattern, pages[pn - 1])
                if m:
                    located = (pn, m.start())
                    res[key] += 1
                    break
        if not located:
            res["page_not_found"] += 1
            if len(res["page_not_found_examples"]) < 6:
                res["page_not_found_examples"].append((ch["document_id"][-10:], ch["page_start"], ch["article_id"], " ".join(words)))
            continue
        pn, pos = located
        before = "\n".join(pages[: pn - 1]) + "\n" + pages[pn - 1][:pos]
        if ch["article_id"]:
            res["article_checked"] += 1
            ids = ID_LINE.findall(before)
            derived = ids[-1] if ids else None
            if derived and related(derived, ch["article_id"]):
                res["article_ok"] += 1
            elif len(res["article_mismatch"]) < 400:
                res["article_mismatch"].append((ch["document_id"][-10:], pn, ch["article_id"], derived, " ".join(words)[:50]))
        body_start = min(c["page_start"] for c in chunks if c["document_id"] == ch["document_id"])
        body_text = "\n".join(pages[body_start - 1 : pn - 1]) + "\n" + pages[pn - 1][:pos]
        in_future = bool(FUTURE_HEAD.search(body_text))
        res["year_checked"] += 1
        if in_future:
            caps = CAPTION.findall(body_text)
            year = None
            if caps:
                last = caps[-1]
                year = int(last[0] or last[1])
            ok = (not ch["is_current"]) and (year == ch["effective_year"] or (year is None and ch["effective_year"] is None))
        else:
            ok = ch["is_current"] and ch["effective_year"] == 2026
        if ok:
            res["year_ok"] += 1
        elif len(res["year_mismatch"]) < 50:
            res["year_mismatch"].append((ch["document_id"][-10:], pn, ch["article_id"], ch["is_current"], ch["effective_year"], in_future))
    return res


def main() -> None:
    client = QdrantClient(path=settings.qdrant_path)
    pts = []
    offset = None
    while True:
        recs, offset = client.scroll(settings.qdrant_collection, limit=500, offset=offset, with_payload=True)
        pts += [{"id": str(p.id), **p.payload} for p in recs]
        if offset is None:
            break
    client.close()
    docs = load_docs()
    out = {"index_wide": audit(pts, docs)}
    details_path = ROOT / "eval" / "results" / "baseline_v1_details.jsonl"
    if details_path.exists():
        retrieved = set()
        for line in details_path.read_text().splitlines():
            d = json.loads(line)
            for m in ("pipeline", "hybrid_nofilter", "dense_nofilter", "bm25_nofilter"):
                for c in d["methods"].get(m, [])[:5]:
                    retrieved.add(c["id"].replace("-", ""))
        subset = [p for p in pts if p["id"].replace("-", "") in retrieved]
        out["retrieved_top5_subset"] = audit(subset, docs)
    (ROOT / "eval" / "results" / "metadata_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        print(f"== {k}: chunks={v['total']}")
        print(f"   page: ok={v['page_ok']} off_by_one={v['page_off_by_one']} not_found={v['page_not_found']} (table/short skipped={v['skipped_table_start']})")
        print(f"   article id vs independent PDF scan: ok={v['article_ok']}/{v['article_checked']}  mismatches={len(v['article_mismatch'])}")
        print(f"   effective year/current flag vs PDF position: ok={v['year_ok']}/{v['year_checked']} mismatches={len(v['year_mismatch'])}")
        print(f"   footer text leaked into chunk: {v['footer_leak']}")
        print("   page_not_found examples:", v["page_not_found_examples"][:3])
        print("   article mismatch examples:", v["article_mismatch"][:6])
        print("   year mismatch examples:", v["year_mismatch"][:4])
        cnt = Counter((m[0], m[2] is None) for m in v["article_mismatch"])


if __name__ == "__main__":
    sys.exit(main())
