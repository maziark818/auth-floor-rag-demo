#!/usr/bin/env python3
"""Run the frozen eval against the demo corpus and print both rankers.

Usage (from the repo root):

    python eval/run_eval.py

Exits 0 when the hybrid ranker clears the MRR gate, 1 otherwise -- the
eval is a gate, not a report nobody reads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluate import MRR_GATE, evaluate_all  # noqa: E402
from app.server import build_index, load_queries  # noqa: E402


def main() -> int:
    index = build_index()
    queries = load_queries()
    report = evaluate_all(index, queries)
    print(f"frozen eval: {len(queries)} queries over {len(index.documents)} documents\n")
    print(f"{'ranker':<8} {'hit@1':>6} {'hit@3':>6} {'mrr@5':>6}")
    for name in ("tfidf", "hybrid"):
        r = report[name]
        print(f"{name:<8} {r['hit@1']:>6.3f} {r['hit@3']:>6.3f} {r['mrr@5']:>6.3f}")
    print("\nper-query (hybrid):")
    for row in report["hybrid"]["per_query"]:
        rank = row["rank"] if row["rank"] is not None else "-"
        print(f"  rank {rank!s:>2}  {row['query']}  -> {row['expected_doc']}")
    mrr = report["hybrid"]["mrr@5"]
    ok = mrr >= MRR_GATE
    print(f"\ngate: hybrid mrr@5 {mrr:.3f} >= {MRR_GATE} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
