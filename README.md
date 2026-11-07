# auth-floor-rag-demo

An MCP server with three things production agent systems actually need and
demo projects usually skip:

1. **An auth floor** — the server *refuses to start* without a token, every
   request is authenticated, and write tools check the caller's tier per call.
2. **A hybrid RAG pipeline** — TF-IDF cosine retrieval with a pool-local BM25
   re-rank fused by Reciprocal Rank Fusion, over a small bundled corpus.
3. **The eval harness that gates it** — a frozen query set, written before
   any tuning, scoring both rankers; the eval exits non-zero below its MRR
   gate, so it can sit in CI.

This is a distilled, sanitized miniature of a real production system (an MCP
control plane fronting a large indexed knowledge store). The architecture,
the auth pattern, and the failure lessons are the production ones; the scale
is deliberately small enough to run anywhere.

## Run it (under 5 minutes)

Requires Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DEMO_MCP_TOKENS='dev-read:read:reader, dev-admin:admin:me'
python -m app.server
# listening on http://127.0.0.1:8765/mcp (8 documents, auth floor ON)
```

Try the floor first:

```bash
curl http://127.0.0.1:8765/health        # public: liveness + data_mode label
curl -X POST http://127.0.0.1:8765/mcp   # 401 — no token, no service
```

And the refusal that defines the floor — start it *without* tokens:

```bash
unset DEMO_MCP_TOKENS
python -m app.server
# FATAL: no MCP token configured ... Refusing to start (exit 2)
```

Connect any MCP client over streamable HTTP with an `Authorization:
Bearer dev-read` header, e.g. a minimal Python client:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    "http://127.0.0.1:8765/mcp",
    headers={"Authorization": "Bearer dev-read"},
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        print(await session.call_tool("search_documents",
                                      {"query": "how does bread rise without commercial yeast"}))
```

## Architecture

```
            Authorization: Bearer <token:tier>
                            │
   MCP client ──HTTP──▶ BearerAuthMiddleware ──401──▶ ✕
                            │  sets principal (contextvar)
                            ▼
                    FastMCP (streamable HTTP)
                    ┌───────────────────────────────┐
                    │ read tools      write tools   │── require admin tier,
                    │ (any token)     (per-call ✓)  │   checked per call
                    └──────────────┬────────────────┘
                                   ▼
                    Index: chunk → TF-IDF cosine pool
                           → pool-local BM25 re-rank → RRF fusion
                                   ▲
                    eval/queries.json (frozen) ── evaluate() scores
                    both rankers; run_eval tool + CLI expose the gate
```

**Tools** (9): `whoami`, `search_documents` (ranker selectable:
`hybrid`/`tfidf`/`bm25`), `get_document`, `list_documents`,
`summarize_document` (extractive), `index_stats`, `run_eval` — plus the
admin-tier writes `add_document` and `remove_document`.

**Auth model.** Tokens are configured as `token:tier[:label]` pairs in
`DEMO_MCP_TOKENS` (or a single admin token in `DEMO_MCP_TOKEN`). Tiers are
`read` < `admin`. Authentication happens in ASGI middleware on *every*
request — the MCP transport never sees an unauthenticated call — and the
resulting principal rides a contextvar so write tools can enforce tier per
call, not once at login. Token comparison is constant-time. `/health` is
public but serves no data, and its payload carries `data_mode:
"sample-corpus"` — see the second lesson below for why that field exists.

**Retrieval.** Documents are chunked paragraph-aware (~700 chars), indexed
as TF-IDF vectors, and queried in two stages: cosine similarity builds a
candidate pool, BM25 computed *over that pool only* re-ranks it, and RRF
fuses the two orderings. TF-IDF is the key-free, deterministic stand-in at
the embedding seam — the interface is where a neural embedder drops in.

## The eval, honestly

`python eval/run_eval.py` scores a frozen set of 16 queries (12 keyword,
4 plain-language paraphrases) against the 8-document demo corpus:

| ranker | hit@1 | hit@3 | MRR@5 |
|---|---|---|---|
| tfidf  | 1.000 | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 | 1.000 |

Perfect scores — because eight documents on eight disjoint topics is an
easy corpus, and pretending otherwise would be exactly the kind of claim
this repo exists to avoid. The demo eval proves the *harness* runs and the
gate bites (lower the threshold in `app/evaluate.py` and watch it fail);
it is not evidence about retrieval quality at scale.

For that, here are the production system's numbers, measured on a frozen
16-query headline set against a ~48,500-chunk snapshot of a 4,406-document
corpus, before and after the fix this demo's hybrid design distills
(pure embedding cosine → + pool-local BM25 re-rank with full-pool RRF):

| production ranker | hit@5 | MRR@5 |
|---|---|---|
| v1: embedding cosine only | 0.6875 | 0.3583 |
| v2.2: + pool-local BM25/RRF | 0.6875 | **0.4135** |

Two honest caveats the production report itself records: the anchored
metrics understate quality (gold chunks were capped at 8 per query while
9–97 were eligible), and on a 3-query broad-control set the anchored
numbers for v2.2 were *worse* even though every control's top-5 stayed
inside the expected topic (topic-level hit@5 was 16/16 headline, 3/3
control). The fix was adopted for the headline gain with the tradeoff
stated, not hidden.

## What broke (the chapter demos usually omit)

**1. The pipe, not the destination.** A remote host appeared unreachable
for hours: the database of "the host is down" evidence kept growing while
the host was fine. The actual fault was a local proxy bridge that read its
input through a *buffered* stream — small client writes were swallowed
before ever reaching the network, so connections half-opened and died.
Fix: pump bytes unbuffered, and add a transport self-test that runs before
any "host down" conclusion is allowed. Lesson: verify the layer you're
standing on before diagnosing the layer you're looking at.

**2. Fixture data wearing live data's clothes.** A dashboard rendered
sample fixtures through the same code path as live data, visually
indistinguishable from the real thing — "it loads" was being read as "it
works." Fix: the payload itself now declares its provenance (`data_mode`
in every health/status response — this demo's `/health` does the same),
so any consumer can tell fixture from live *in-band*, without trusting
convention or memory. Lesson: test doubles must be labeled in the data,
not in the documentation.

**3. Green by inspection isn't green.** A large integration branch looked
mergeable on review. Running the full suite in a clean environment found
exactly one genuine defect — a missing dependency — hiding among failures
that turned out to be environment artifacts (a VM clock-resolution quirk
making two timing assertions land on exactly 0.0 ms; a test-runner quirk
in one sandbox). Both kinds of red look identical in a summary line. The
rule that came out of it: no merge without a recorded green run from a
clean checkout, and an "environment artifact" must be *proven* — reproduced
and root-caused — never assumed. This repo follows the rule: the test
suite spins up the real server process and exercises the floor over HTTP,
because the floor only means something at the transport.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest        # 19 tests: auth units, retrieval units,
                        # and end-to-end HTTP tests against the real
                        # server process (401s, tier denial, no-start)
python eval/run_eval.py # the frozen eval + gate
```

## Layout

```
app/auth.py       token store, tiers, ASGI bearer middleware, no-start floor
app/rag.py        chunking, TF-IDF + pool-local BM25, RRF hybrid search
app/evaluate.py   frozen-eval scoring (hit@1/@3, MRR@5) + the gate
app/server.py     FastMCP wiring: 9 tools, /health, startup
corpus/           8 sample documents (original text, written for this demo)
eval/queries.json the frozen query set · eval/run_eval.py the CLI gate
tests/            unit + live-transport end-to-end tests
```

## License

MIT. The corpus documents are original texts written for this demo.
