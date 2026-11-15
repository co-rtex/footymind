"""Request limits for a publicly reachable server (pure ASGI middleware).

Two independent guards, both configurable through the environment:

* **Per-client rate limit** on ``/api`` requests: a token bucket per client address
  (``FOOTYMIND_RATE_LIMIT_PER_MIN``, default 120, with a burst of the same size; 0 disables it).
  Over the limit the server answers 429 with ``Retry-After``. Behind a reverse proxy every
  request arrives from the proxy's address, so set ``FOOTYMIND_CLIENT_IP_HEADER`` to the header
  the proxy fills with the client address (e.g. ``X-Real-IP``); without it all clients share one
  bucket, which still caps the total load. Loopback clients (local development, tests, the
  container's own health check) and ``/api/health`` are never limited.
* **Concurrency cap** on the expensive endpoints (scenarios, frame analysis, Defensive Shadow,
  similar moments): at most ``FOOTYMIND_MAX_HEAVY_REQUESTS`` (default 4) run at once; others
  wait up to ``FOOTYMIND_HEAVY_WAIT_S`` (default 15 s) and then get 503 with ``Retry-After``.

These protect a small instance and its hosting allowance from accidental or deliberate floods;
they are not a substitute for an edge rate limiter on a production service.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from typing import Any

from fastapi.responses import JSONResponse

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
HEAVY = re.compile(
    r"^/api/(scenarios$|matches/[^/]+/frames/[^/]+/(analysis|defensive-shadow|similar)$)"
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class RequestLimits:
    def __init__(
        self,
        app: Any,
        per_min: float | None = None,
        max_heavy: int | None = None,
        heavy_wait_s: float | None = None,
        client_ip_header: str | None = None,
    ) -> None:
        self.app = app
        self.per_min = (
            _env_float("FOOTYMIND_RATE_LIMIT_PER_MIN", 120) if per_min is None else per_min
        )
        self.capacity = max(1.0, self.per_min)
        self.max_heavy = (
            int(_env_float("FOOTYMIND_MAX_HEAVY_REQUESTS", 4)) if max_heavy is None else max_heavy
        )
        self.heavy_wait_s = (
            _env_float("FOOTYMIND_HEAVY_WAIT_S", 15) if heavy_wait_s is None else heavy_wait_s
        )
        header = (
            os.environ.get("FOOTYMIND_CLIENT_IP_HEADER", "")
            if client_ip_header is None
            else client_ip_header
        )
        self.header = header.strip().lower().encode() or None
        self.buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, last time)
        self.heavy: asyncio.Semaphore | None = None

    def client(self, scope: Any) -> str:
        if self.header is not None:
            # the proxy's own value comes last if a client also sent the header
            values = [v for k, v in scope.get("headers") or [] if k == self.header]
            if values:
                last = values[-1].decode("latin-1").split(",")[-1].strip()
                if last:
                    return last
        peer = scope.get("client")
        return str(peer[0]) if peer else "unknown"

    def take(self, key: str, now: float) -> float:
        """Spend one token; return 0 if allowed, else the seconds until one is available."""
        rate = self.per_min / 60.0
        tokens, last = self.buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * rate)
        if tokens >= 1.0:
            self.buckets[key] = (tokens - 1.0, now)
            return 0.0
        self.buckets[key] = (tokens, now)
        return (1.0 - tokens) / rate

    def prune(self, now: float) -> None:
        if len(self.buckets) > 10_000:  # forget clients idle long enough to be full again
            full_after = self.capacity / (self.per_min / 60.0)
            self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < full_after}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if not path.startswith("/api/") or path == "/api/health":  # health checks: never limited
            await self.app(scope, receive, send)
            return
        key = self.client(scope)
        if self.per_min > 0 and key not in LOOPBACK:
            now = time.monotonic()
            self.prune(now)
            wait = self.take(key, now)
            if wait > 0:
                retry = str(max(1, math.ceil(wait)))
                await JSONResponse(
                    status_code=429,
                    content={"detail": "too many requests; please slow down"},
                    headers={"Retry-After": retry},
                )(scope, receive, send)
                return
        if self.max_heavy <= 0 or not HEAVY.match(path):
            await self.app(scope, receive, send)
            return
        if self.heavy is None:  # created inside the running event loop
            self.heavy = asyncio.Semaphore(self.max_heavy)
        try:
            await asyncio.wait_for(self.heavy.acquire(), timeout=self.heavy_wait_s)
        except TimeoutError:
            await JSONResponse(
                status_code=503,
                content={"detail": "the server is busy; please retry in a few seconds"},
                headers={"Retry-After": "5"},
            )(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.heavy.release()
