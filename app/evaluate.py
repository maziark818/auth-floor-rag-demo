"""The frozen-eval harness: the gate made visible.

A frozen query set (``eval/queries.json``, written before any tuning) with
the document each query should find. Both rankers are scored on the same
set: hit@1, hit@3, and MRR@5 at the *document* level (first occurrence of
the expected document in the ranked chunk list). The CLI exits non-zero if
the hybrid ranker falls below the MRR gate, so the eval can sit in CI the
same way the production system's frozen eval gates retrieval changes.
"""

from __future__ import annotations

from .rag import Index

MRR_GATE = 0.5


def evaluate(index: Index, queries: list[dict], ranker: str) -> dict:
    per_query = []
    rr_sum = 0.0
    hit1 = 0
    hit3 = 0
    for q in queries:
        results = index.search(q["query"], k=5, ranker=ranker)
        rank = None
        for i, r in enumerate(results):
            if r.doc_id == q["expected_doc"]:
                rank = i + 1
                break
        rr = 1.0 / rank if rank else 0.0
        rr_sum += rr
        hit1 += 1 if rank == 1 else 0
        hit3 += 1 if rank is not None and rank <= 3 else 0
        per_query.append(
            {"query": q["query"], "expected_doc": q["expected_doc"], "rank": rank}
        )
    n = max(len(queries), 1)
    return {
        "ranker": ranker,
        "queries": len(queries),
        "hit@1": round(hit1 / n, 4),
        "hit@3": round(hit3 / n, 4),
        "mrr@5": round(rr_sum / n, 4),
        "per_query": per_query,
    }


def evaluate_all(index: Index, queries: list[dict]) -> dict:
    return {
        "tfidf": evaluate(index, queries, "tfidf"),
        "hybrid": evaluate(index, queries, "hybrid"),
        "gate": {"metric": "hybrid mrr@5", "threshold": MRR_GATE},
    }
