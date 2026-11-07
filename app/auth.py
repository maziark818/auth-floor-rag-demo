"""Token auth floor for the demo MCP server.

The pattern, distilled from a production MCP control plane:

* **No token configured -> the server refuses to start.** There is no
  anonymous mode, no default token, and no fallback branch.
* **Every HTTP request is authenticated** by ASGI middleware before it
  reaches the MCP transport: ``Authorization: Bearer <token>`` or 401.
* **Tokens carry a tier** (``read`` < ``admin``). Read tools accept any
  valid token; write tools additionally require the admin tier, and that
  check runs *per call*, against the principal who made the request --
  not once at startup.
"""

from __future__ import annotations

import hmac
import os
import sys
from contextvars import ContextVar
from dataclasses import dataclass

TIER_READ = "read"
TIER_ADMIN = "admin"
_TIER_RANK = {TIER_READ: 1, TIER_ADMIN: 2}


class AuthConfigError(RuntimeError):
    """Raised when no usable token configuration exists."""


@dataclass(frozen=True)
class Principal:
    tier: str
    label: str


# Set by the middleware for the duration of each authenticated request.
current_principal: ContextVar[Principal | None] = ContextVar(
    "current_principal", default=None
)


class TokenStore:
    """Maps bearer tokens to principals."""

    def __init__(self, entries: dict[str, Principal]):
        if not entries:
            raise AuthConfigError("token store is empty")
        self._entries = dict(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def authenticate(self, authorization_header: str | None) -> Principal | None:
        """Return the Principal for a valid ``Bearer`` header, else None."""
        if not authorization_header:
            return None
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        token = token.strip()
        # Constant-time comparisons against every configured token.
        matched: Principal | None = None
        for known, principal in self._entries.items():
            if hmac.compare_digest(token, known):
                matched = principal
        return matched

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "TokenStore":
        """Build a store from the environment.

        ``DEMO_MCP_TOKENS``: comma-separated ``token:tier[:label]`` entries,
        e.g. ``DEMO_MCP_TOKENS="s3cret-read:read:ci, s3cret-admin:admin:me"``.

        ``DEMO_MCP_TOKEN``: convenience single token, tier ``admin``.
        """
        env = dict(os.environ if env is None else env)
        entries: dict[str, Principal] = {}
        raw = env.get("DEMO_MCP_TOKENS", "")
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            fields = [f.strip() for f in part.split(":")]
            if len(fields) < 2 or not fields[0] or fields[1] not in _TIER_RANK:
                raise AuthConfigError(
                    f"malformed DEMO_MCP_TOKENS entry {part!r}; "
                    "expected token:tier[:label] with tier read|admin"
                )
            token, tier = fields[0], fields[1]
            label = fields[2] if len(fields) > 2 and fields[2] else tier
            entries[token] = Principal(tier=tier, label=label)
        single = env.get("DEMO_MCP_TOKEN", "").strip()
        if single:
            entries.setdefault(single, Principal(tier=TIER_ADMIN, label="default"))
        if not entries:
            raise AuthConfigError(
                "no MCP token configured: set DEMO_MCP_TOKENS (or DEMO_MCP_TOKEN)"
            )
        return cls(entries)


def tier_allows(principal: Principal, required: str) -> bool:
    return _TIER_RANK[principal.tier] >= _TIER_RANK[required]


def require_tier(required: str) -> Principal:
    """Per-call tier check used by write tools. Raises PermissionError."""
    principal = current_principal.get()
    if principal is None:
        raise PermissionError("unauthenticated: no principal on this request")
    if not tier_allows(principal, required):
        raise PermissionError(
            f"token tier {principal.tier!r} may not perform this operation; "
            f"requires tier {required!r}"
        )
    return principal


def load_store_or_exit() -> TokenStore:
    """The floor itself: unconfigured means the process does not run."""
    try:
        return TokenStore.from_env()
    except AuthConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        print(
            "Refusing to start: this server has no anonymous mode. "
            "Example: DEMO_MCP_TOKENS='dev-read:read:reader, dev-admin:admin:me'",
            file=sys.stderr,
        )
        sys.exit(2)


class BearerAuthMiddleware:
    """ASGI middleware enforcing the floor on every HTTP request.

    ``/health`` is public (liveness only, no data); everything else needs a
    valid bearer token. The authenticated principal is published on a
    contextvar so tool handlers can run per-call tier checks.
    """

    def __init__(self, app, store: TokenStore, health_payload: dict | None = None):
        self.app = app
        self.store = store
        self.health_payload = health_payload or {"status": "ok"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == "/health":
            from starlette.responses import JSONResponse

            await JSONResponse(self.health_payload)(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        principal = self.store.authenticate(headers.get("authorization"))
        if principal is None:
            from starlette.responses import JSONResponse

            await JSONResponse(
                {"error": "unauthorized", "detail": "valid bearer token required"},
                status_code=401,
            )(scope, receive, send)
            return
        token = current_principal.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            current_principal.reset(token)
