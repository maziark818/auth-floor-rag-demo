import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.auth import (
    AuthConfigError,
    BearerAuthMiddleware,
    Principal,
    TokenStore,
    require_tier,
    current_principal,
    tier_allows,
)


def test_from_env_parses_tiers_and_labels():
    store = TokenStore.from_env(
        {"DEMO_MCP_TOKENS": "tok-read:read:ci, tok-admin:admin:me"}
    )
    assert len(store) == 2
    assert store.authenticate("Bearer tok-read") == Principal("read", "ci")
    assert store.authenticate("Bearer tok-admin") == Principal("admin", "me")


def test_from_env_single_token_convenience_is_admin():
    store = TokenStore.from_env({"DEMO_MCP_TOKEN": "solo"})
    assert store.authenticate("Bearer solo").tier == "admin"


def test_from_env_requires_some_token():
    with pytest.raises(AuthConfigError):
        TokenStore.from_env({})
    with pytest.raises(AuthConfigError):
        TokenStore.from_env({"DEMO_MCP_TOKENS": "  , "})


def test_from_env_rejects_malformed_entries():
    with pytest.raises(AuthConfigError):
        TokenStore.from_env({"DEMO_MCP_TOKENS": "tok:superuser"})
    with pytest.raises(AuthConfigError):
        TokenStore.from_env({"DEMO_MCP_TOKENS": "justatoken"})


def test_authenticate_rejects_bad_headers():
    store = TokenStore.from_env({"DEMO_MCP_TOKENS": "tok:read"})
    assert store.authenticate(None) is None
    assert store.authenticate("") is None
    assert store.authenticate("Basic tok") is None
    assert store.authenticate("Bearer wrong") is None
    assert store.authenticate("Bearer") is None


def test_tier_ordering():
    assert tier_allows(Principal("admin", "x"), "read")
    assert tier_allows(Principal("admin", "x"), "admin")
    assert tier_allows(Principal("read", "x"), "read")
    assert not tier_allows(Principal("read", "x"), "admin")


def test_require_tier_uses_current_principal():
    token = current_principal.set(Principal("read", "ci"))
    try:
        assert require_tier("read").label == "ci"
        with pytest.raises(PermissionError):
            require_tier("admin")
    finally:
        current_principal.reset(token)


def _client(store):
    async def home(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/data", home)])
    return TestClient(
        BearerAuthMiddleware(app, store, health_payload={"status": "ok"}),
        raise_server_exceptions=False,
    )


def test_middleware_health_is_public_and_data_is_not():
    client = _client(TokenStore.from_env({"DEMO_MCP_TOKENS": "tok:read"}))
    assert client.get("/health").status_code == 200
    assert client.get("/data").status_code == 401
    assert client.get("/data", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/data", headers={"Authorization": "Bearer tok"}).status_code == 200
