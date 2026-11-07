"""A small, dependency-free retrieval pipeline.

Design mirrors the production lesson this demo is distilled from (see the
README): a single ranking signal is brittle, so retrieval here is a
**hybrid** -- TF-IDF cosine over chunks produces a candidate pool, then
BM25 re-ranks that pool and the two orderings are fused with Reciprocal
Rank Fusion (RRF). The demo corpus is tiny, so the effect is modest; the
*shape* is the point, and the frozen eval (``app.evaluate``) measures both
rankers so the claim is checkable rather than asserted.

Embeddings: TF-IDF vectors are the key-free, offline "embedding" -- no API
key, no model download, fully deterministic. The interface (``Index.search``)
is the seam where a neural embedder would slot in.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the to was what when where which who why with".split()
)


def tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    ]


def chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunking with a hard size cap and small overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :]
        buf = para
    if buf:
        chunks.append(buf)
    return chunks


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    doc_id: str
    chunk_index: int
    score: float
    snippet: str


class Index:
    """Chunk store + TF-IDF / BM25 statistics, built incrementally."""

    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}  # doc_id -> {title, text}
        self.chunks: list[Chunk] = []
        self._df: Counter = Counter()  # token -> # chunks containing it
        self._doc_len: list[int] = []

    # -- ingestion ------------------------------------------------------
    def add_document(self, doc_id: str, title: str, text: str) -> int:
        if doc_id in self.documents:
            self.remove_document(doc_id)
        pieces = chunk_text(text)
        self.documents[doc_id] = {"title": title, "text": text}
        for i, piece in enumerate(pieces):
            tokens = tokenize(piece)
            chunk = Chunk(doc_id=doc_id, chunk_index=i, text=piece, tokens=tokens)
            self.chunks.append(chunk)
            self._doc_len.append(len(tokens))
            for tok in set(tokens):
                self._df[tok] += 1
        return len(pieces)

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        kept: list[Chunk] = []
        kept_len: list[int] = []
        for chunk, ln in zip(self.chunks, self._doc_len):
            if chunk.doc_id == doc_id:
                for tok in set(chunk.tokens):
                    self._df[tok] -= 1
                    if self._df[tok] <= 0:
                        del self._df[tok]
            else:
                kept.append(chunk)
                kept_len.append(ln)
        self.chunks, self._doc_len = kept, kept_len
        del self.documents[doc_id]
        return True

    # -- scoring --------------------------------------------------------
    def _idf(self, token: str) -> float:
        n = max(len(self.chunks), 1)
        return math.log(1 + (n - self._df.get(token, 0) + 0.5) / (self._df.get(token, 0) + 0.5))

    def _tfidf_cosine(self, query_tokens: list[str]) -> dict[int, float]:
        if not self.chunks:
            return {}
        q_counts = Counter(query_tokens)
        q_vec = {t: c * self._idf(t) for t, c in q_counts.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scores: dict[int, float] = {}
        for idx, chunk in enumerate(self.chunks):
            c_counts = Counter(chunk.tokens)
            dot = sum(q_vec[t] * c_counts.get(t, 0) * self._idf(t) for t in q_vec)
            if dot <= 0:
                continue
            c_norm = math.sqrt(
                sum((c * self._idf(t)) ** 2 for t, c in c_counts.items())
            ) or 1.0
            scores[idx] = dot / (q_norm * c_norm)
        return scores

    def _bm25(self, query_tokens: list[str], pool: list[int]) -> dict[int, float]:
        """BM25 computed over the candidate pool only (pool-local stats are
        the production fix this demo mirrors: corpus-wide lexical stats let
        term-dense boilerplate crowd out the chunk that states the answer)."""
        if not pool:
            return {}
        pool_df: Counter = Counter()
        for idx in pool:
            for tok in set(self.chunks[idx].tokens):
                pool_df[tok] += 1
        n = len(pool)
        avgdl = (sum(self._doc_len[i] for i in pool) / n) if n else 1.0
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for idx in pool:
            chunk = self.chunks[idx]
            counts = Counter(chunk.tokens)
            dl = self._doc_len[idx] or 1
            score = 0.0
            for tok in set(query_tokens):
                if tok not in counts:
                    continue
                idf = math.log(1 + (n - pool_df[tok] + 0.5) / (pool_df[tok] + 0.5))
                tf = counts[tok]
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            if score > 0:
                scores[idx] = score
        return scores

    # -- search ---------------------------------------------------------
    def search(
        self, query: str, k: int = 5, ranker: str = "hybrid", pool_size: int = 20
    ) -> list[SearchResult]:
        """Rank chunks. ranker: 'tfidf' | 'bm25' | 'hybrid' (RRF fusion)."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        cosine = self._tfidf_cosine(query_tokens)
        pool = [
            idx
            for idx, _ in sorted(cosine.items(), key=lambda kv: -kv[1])[:pool_size]
        ]
        if ranker == "tfidf":
            ranked = sorted(cosine.items(), key=lambda kv: -kv[1])
        else:
            bm25 = self._bm25(query_tokens, pool or list(range(len(self.chunks))))
            if ranker == "bm25":
                ranked = sorted(bm25.items(), key=lambda kv: -kv[1])
            elif ranker == "hybrid":
                fused: dict[int, float] = defaultdict(float)
                for rank, (idx, _) in enumerate(
                    sorted(cosine.items(), key=lambda kv: -kv[1])
                ):
                    fused[idx] += 1.0 / (60 + rank + 1)
                for rank, (idx, _) in enumerate(
                    sorted(bm25.items(), key=lambda kv: -kv[1])
                ):
                    fused[idx] += 1.0 / (60 + rank + 1)
                ranked = sorted(fused.items(), key=lambda kv: -kv[1])
            else:
                raise ValueError(f"unknown ranker {ranker!r}")
        results: list[SearchResult] = []
        for idx, score in ranked[:k]:
            chunk = self.chunks[idx]
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            results.append(
                SearchResult(
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    score=round(float(score), 6),
                    snippet=snippet,
                )
            )
        return results

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vocabulary": len(self._df),
        }


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Pick the sentences densest in the document's own frequent terms."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if len(s.strip()) > 30]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    freq = Counter(tokenize(text))
    scored = sorted(
        range(len(sentences)),
        key=lambda i: -sum(freq[t] for t in tokenize(sentences[i])),
    )
    chosen = sorted(scored[:max_sentences])
    return " ".join(sentences[i] for i in chosen)
