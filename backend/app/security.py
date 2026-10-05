"""API security helpers: API-key guard, rate limiter, security headers."""
from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Guard for state-changing routes. Disabled when ``PULSE_API_KEY`` is unset (demo mode)."""
    expected = request.app.state.container.settings.api_key
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


class RateLimiter:
    """In-memory sliding-window limiter keyed by client IP (single-process; put a gateway in
    front for multi-replica deployments)."""

    def __init__(self, max_calls: int, per_seconds: float, name: str):
        self.max_calls = max_calls
        self.per = per_seconds
        self.name = name
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > self.per:
            q.popleft()
        if len(q) >= self.max_calls:
            raise HTTPException(status_code=429, detail=f"rate limit exceeded for {self.name}", headers={"Retry-After": str(int(self.per))})
        q.append(now)

    def reset(self) -> None:
        self._hits.clear()


limit_classify = RateLimiter(300, 60, "classification")
limit_compute = RateLimiter(120, 60, "simulation")
limit_ingest = RateLimiter(30, 60, "ingestion")
limit_jobs = RateLimiter(30, 60, "background jobs")
ALL_LIMITERS = [limit_classify, limit_compute, limit_ingest, limit_jobs]

_DOC_PATHS = ("/docs", "/redoc", "/openapi.json")
_APP_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self' ws: wss:; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if request.url.path.startswith("/api") and not request.url.path.startswith(_DOC_PATHS):
            resp.headers.setdefault("Cache-Control", "no-store")
            resp.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        elif not request.url.path.startswith(_DOC_PATHS):  # the single-page app (when served by this process)
            resp.headers.setdefault("Content-Security-Policy", _APP_CSP)
        return resp


__all__ = ["require_api_key", "RateLimiter", "SecurityHeaders", "Depends", "ALL_LIMITERS"]
