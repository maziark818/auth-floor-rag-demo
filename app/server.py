"""Demo MCP server: an auth floor + a hybrid RAG pipeline + its own eval.

Run (from the repo root):

    DEMO_MCP_TOKENS='dev-read:read:reader, dev-admin:admin:me' \\
        python -m app.server

With no token configured the process exits before binding a socket --
that refusal *is* the auth floor. Transport is streamable HTTP (the same
shape as the production system this is distilled from): every request
carries its bearer token, so authentication and per-call tier checks
happen per request, not once at launch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP

from .auth import (
    TIER_ADMIN,
    TIER_READ,
    BearerAuthMiddleware,
    current_principal,
    load_store_or_exit,
    require_tier,
)
from .evaluate import evaluate_all
from .rag import Index

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
QUERIES_PATH = ROOT / "eval" / "queries.json"
HOST = os.environ.get("DEMO_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEMO_MCP_PORT", "8765"))
DATA_MODE = "sample-corpus"  # labeled in-band: this is never production data


def build_index() -> Index:
    index = Index()
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.lstrip("# ").splitlines()[0].strip() if text.strip() else path.stem
        index.add_document(path.stem, title, text)
    return index


def load_queries() -> list[dict]:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]


def build_server(index: Index) -> FastMCP:
    mcp = FastMCP("auth-floor-rag-demo", host=HOST, port=PORT)

    @mcp.tool()
    def whoami() -> dict:
        """Return the tier and label of the token making this call."""
        principal = current_principal.get()
        if principal is None:
            return {"authenticated": False}
        return {
            "authenticated": True,
            "tier": principal.tier,
            "label": principal.label,
        }

    @mcp.tool()
    def search_documents(query: str, k: int = 5, ranker: str = "hybrid") -> list[dict]:
        """Search the document corpus. ranker: 'hybrid' (default), 'tfidf', 'bm25'."""
        return [
            {
                "doc_id": r.doc_id,
                "chunk": r.chunk_index,
                "score": r.score,
                "snippet": r.snippet,
            }
            for r in index.search(query, k=k, ranker=ranker)
        ]

    @mcp.tool()
    def get_document(doc_id: str) -> dict:
        """Fetch a full document by id."""
        doc = index.documents.get(doc_id)
        if doc is None:
            return {"error": f"no document {doc_id!r}"}
        return {"doc_id": doc_id, "title": doc["title"], "text": doc["text"]}

    @mcp.tool()
    def list_documents() -> list[dict]:
        """List every document in the corpus with its chunk count."""
        counts: dict[str, int] = {}
        for chunk in index.chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return [
            {"doc_id": d, "title": meta["title"], "chunks": counts.get(d, 0)}
            for d, meta in sorted(index.documents.items())
        ]

    @mcp.tool()
    def summarize_document(doc_id: str) -> dict:
        """Extractive summary: the document's own most term-dense sentences."""
        from .rag import extractive_summary

        doc = index.documents.get(doc_id)
        if doc is None:
            return {"error": f"no document {doc_id!r}"}
        return {"doc_id": doc_id, "summary": extractive_summary(doc["text"])}

    @mcp.tool()
    def index_stats() -> dict:
        """Corpus size, chunk count, vocabulary size, and the data mode."""
        return {**index.stats(), "data_mode": DATA_MODE}

    @mcp.tool()
    def run_eval() -> dict:
        """Run the frozen eval against the live index (both rankers)."""
        return evaluate_all(index, load_queries())

    @mcp.tool()
    def add_document(doc_id: str, title: str, text: str) -> dict:
        """Add a document to the index. Requires the admin tier."""
        require_tier(TIER_ADMIN)
        chunks = index.add_document(doc_id, title, text)
        return {"doc_id": doc_id, "chunks_added": chunks}

    @mcp.tool()
    def remove_document(doc_id: str) -> dict:
        """Remove a document from the index. Requires the admin tier."""
        require_tier(TIER_ADMIN)
        return {"doc_id": doc_id, "removed": index.remove_document(doc_id)}

    return mcp


def main() -> None:
    store = load_store_or_exit()  # exits(2) when unconfigured: the floor
    index = build_index()
    mcp = build_server(index)
    health = {
        "status": "ok",
        "auth_enabled": True,
        "tokens_configured": len(store),
        "documents": len(index.documents),
        "data_mode": DATA_MODE,
    }
    app = BearerAuthMiddleware(
        mcp.streamable_http_app(), store, health_payload=health
    )
    print(
        f"listening on http://{HOST}:{PORT}/mcp "
        f"({len(index.documents)} documents, auth floor ON)",
        file=sys.stderr,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
