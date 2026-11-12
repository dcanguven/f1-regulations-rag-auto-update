from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_lib import Workbench, settings
from app.retrieval.context_expander import expand_context
from experiment_lib import has_evidence, is_relevant


def row(m):
    return (
        f"art R@1={m['art@1']:.3f} R@3={m['art@3']:.3f} R@5={m['art@5']:.3f} | "
        f"ev R@1={m['ev@1']:.3f} R@5={m['ev@5']:.3f} | MRR={m['mrr']:.3f}"
    )


def main() -> None:
    wb = Workbench()
    non_exact = wb.non_exact_ids()
    base = wb.sparse_lists(wb.bm25("base"))
    print(f"questions={len(wb.questions)} exact-lookup={len(wb.exact)} non-exact answerable={len(non_exact)}")
    print("production composite n=54:", row(wb.score(wb.fused(base))))
    print("production non-exact:", row(wb.score(wb.fused(base), only=non_exact)))
    print("dense only non-exact:", row(wb.score({q['id']: wb.dense[q['id']][:5] for q in wb.questions}, only=non_exact)))
    print("bm25 only non-exact:", row(wb.score({q['id']: base[q['id']][:5] for q in wb.questions}, only=non_exact)))
    print("\nFUSION GRID (rrf_k, w_dense, pool_dense, pool_bm25), non-exact subset")
    grid = []
    for k in (10, 30, 60):
        for wd in (0.5, 0.75, 1.0, 1.5, 2.0):
            for pd in (10, 20, 50):
                for ps in (10, 20, 50):
                    grid.append(((k, wd, pd, ps), wb.score(wb.fused(base, k=k, w_dense=wd, pool_d=pd, pool_s=ps), only=non_exact)))
    grid.sort(key=lambda x: -(x[1]["art@5"] + x[1]["ev@5"] + x[1]["mrr"]))
    for cfg, m in grid[:8]:
        print(cfg, row(m))
    for key in ("art@5", "ev@5", "mrr"):
        vals = [m[key] for _, m in grid]
        print(f"{key}: min {min(vals):.3f} median {statistics.median(vals):.3f} max {max(vals):.3f}")
    print("\nBM25 VARIANTS (non-exact subset): BM25 alone | hybrid")
    for tok in ("base", "nostop", "stem", "nostop_stem"):
        for k1, b in ((1.5, 0.75), (1.2, 0.75), (1.5, 0.5), (2.0, 0.75), (1.2, 0.9)):
            sp = wb.sparse_lists(wb.bm25(tok, k1, b))
            print(f"{tok:12} k1={k1} b={b}: bm25 {row(wb.score({q['id']: sp[q['id']][:5] for q in wb.questions}, only=non_exact))}")
            print(f"{'':26} hybrid {row(wb.score(wb.fused(sp), only=non_exact))}")
    print("\nCONTEXT EXPANSION")
    fused = wb.fused(base)
    added, rescued_art, rescued_ev, times = 0, [], [], []
    for q in wb.questions:
        top = fused[q["id"]]
        t = time.perf_counter()
        expanded = expand_context(top, wb.client, settings.qdrant_collection)
        times.append((time.perf_counter() - t) * 1000)
        extra = expanded[len(top):]
        added += len(extra)
        if not q["answerable"] or not extra:
            continue
        if not any(is_relevant(c["payload"], q) for c in top) and any(is_relevant(c["payload"], q) for c in extra):
            rescued_art.append(q["id"])
        if not any(has_evidence(c["payload"], q) for c in top) and any(has_evidence(c["payload"], q) for c in extra):
            rescued_ev.append(q["id"])
    print(f"chunks added over {len(times)} questions: {added}; article rescues: {rescued_art}; evidence rescues: {rescued_ev}")
    print(f"expand_context median {statistics.median(times):.1f} ms, p95 {sorted(times)[int(0.95 * (len(times) - 1))]:.1f} ms")


if __name__ == "__main__":
    main()
