"""End-to-end tests against the real server process over HTTP.

These exist because the auth floor only means something at the transport:
no token configured -> the process exits; no/invalid bearer -> 401 before
the MCP layer; valid read token -> tools answer; read token attempting a
write -> the per-call tier check refuses.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parent.parent
READ = "test-read-token"
ADMIN = "test-admin-token"

# Some CI/VM environments export a no_proxy list that httpx cannot parse
# (e.g. bracketed IPv6 entries); loopback traffic must bypass proxies
# anyway, so pin a minimal, parseable value for this test process.
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DEMO_MCP") and k != "DEMO_MCP_TOKEN"
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "DEMO_MCP_TOKENS": f"{READ}:read:reader, {ADMIN}:admin:admin",
            "DEMO_MCP_PORT": str(port),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early: {proc.stderr.read()}")
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError("server did not become healthy in 30s")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


def _call(base: str, token: str, fn):
    async def _run():
        headers = {"Authorization": f"Bearer {token}"}
        async with streamablehttp_client(f"{base}/mcp", headers=headers) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    return asyncio.run(_run())


def _payload(result):
    assert not result.isError, result.content
    # Prefer structured content: for list-typed tools FastMCP emits one
    # text block per item, while structuredContent carries the whole list
    # (wrapped as {"result": [...]}).
    if result.structuredContent is not None:
        data = result.structuredContent
        if isinstance(data, dict) and set(data) == {"result"}:
            return data["result"]
        return data
    return json.loads(result.content[0].text)


def test_health_is_public_and_labels_its_data(server):
    r = httpx.get(f"{server}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_enabled"] is True
    assert body["data_mode"] == "sample-corpus"
    assert body["documents"] == 8


def test_mcp_endpoint_rejects_missing_and_wrong_tokens(server):
    assert httpx.post(f"{server}/mcp", json={}).status_code == 401
    r = httpx.post(
        f"{server}/mcp", json={}, headers={"Authorization": "Bearer wrong"}
    )
    assert r.status_code == 401


def test_read_token_lists_tools_and_searches(server):
    async def work(session):
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert {
            "search_documents",
            "get_document",
            "list_documents",
            "summarize_document",
            "index_stats",
            "run_eval",
            "whoami",
            "add_document",
            "remove_document",
        } <= names
        me = _payload(await session.call_tool("whoami", {}))
        assert me == {"authenticated": True, "tier": "read", "label": "reader"}
        hits = _payload(
            await session.call_tool(
                "search_documents", {"query": "movable type Gutenberg"}
            )
        )
        assert hits and hits[0]["doc_id"] == "printing-press"
        ev = _payload(await session.call_tool("run_eval", {}))
        assert ev["hybrid"]["mrr@5"] >= 0.5
        return True

    assert _call(server, READ, work)


def test_read_token_cannot_write_but_admin_can(server):
    async def denied(session):
        result = await session.call_tool(
            "add_document",
            {"doc_id": "x", "title": "X", "text": "unauthorized write attempt"},
        )
        assert result.isError
        assert "admin" in result.content[0].text

    _call(server, READ, denied)

    async def allowed(session):
        added = _payload(
            await session.call_tool(
                "add_document",
                {
                    "doc_id": "zephyr",
                    "title": "Zephyr Winds",
                    "text": "A zephyr is a soft gentle west wind. " * 4,
                },
            )
        )
        assert added["chunks_added"] >= 1
        hits = _payload(
            await session.call_tool("search_documents", {"query": "zephyr west wind"})
        )
        assert hits and hits[0]["doc_id"] == "zephyr"
        removed = _payload(
            await session.call_tool("remove_document", {"doc_id": "zephyr"})
        )
        assert removed["removed"] is True

    _call(server, ADMIN, allowed)


def test_no_token_configured_means_no_start():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DEMO_MCP") and k != "DEMO_MCP_TOKEN"
    }
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "app.server"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    assert "Refusing to start" in proc.stderr
