"""Hardening for a publicly reachable server (production audit, 2026-11-15). No data needed.

Each test builds a tiny FastAPI app around the real middleware or static-file class, so the
behaviour is tested in isolation from match data.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from footymind.api.limits import RequestLimits


def _app(**limits: object) -> FastAPI:
    a = FastAPI()

    @a.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @a.get("/api/matches")
    def matches() -> list[str]:
        return []

    @a.post("/api/scenarios")
    async def scenario() -> dict[str, bool]:
        await asyncio.sleep(0.3)  # a slow, "heavy" computation
        return {"ok": True}

    a.add_middleware(RequestLimits, **limits)  # type: ignore[arg-type]
    return a


def test_per_client_rate_limit_answers_429_with_retry_after() -> None:
    app = _app(per_min=3, max_heavy=0, client_ip_header="")
    c = TestClient(app, client=("203.0.113.5", 1234))
    assert [c.get("/api/matches").status_code for _ in range(3)] == [200, 200, 200]
    r = c.get("/api/matches")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # another client has its own bucket; health checks and loopback clients are never limited
    assert TestClient(app, client=("203.0.113.6", 1)).get("/api/matches").status_code == 200
    assert all(c.get("/api/health").status_code == 200 for _ in range(10))
    local = TestClient(app, client=("127.0.0.1", 1))
    assert all(local.get("/api/matches").status_code == 200 for _ in range(10))


def test_client_ip_header_keys_the_bucket_and_the_proxys_value_wins() -> None:
    app = _app(per_min=2, max_heavy=0, client_ip_header="X-Real-IP")
    proxy = TestClient(app, client=("10.0.0.1", 1))  # every request arrives from the proxy
    for _ in range(2):
        assert proxy.get("/api/matches", headers={"X-Real-IP": "198.51.100.1"}).status_code == 200
    assert proxy.get("/api/matches", headers={"X-Real-IP": "198.51.100.1"}).status_code == 429
    assert proxy.get("/api/matches", headers={"X-Real-IP": "198.51.100.2"}).status_code == 200
    # a client-sent value cannot pick a fresh bucket: the last (proxy-written) value is used
    spoof = [("X-Real-IP", "192.0.2.99"), ("X-Real-IP", "198.51.100.1")]
    assert proxy.get("/api/matches", headers=httpx.Headers(spoof)).status_code == 429


def test_heavy_requests_beyond_the_cap_wait_then_get_503() -> None:
    app = _app(per_min=0, max_heavy=1, heavy_wait_s=0.1)

    async def run() -> list[int]:
        transport = httpx.ASGITransport(app=app, client=("203.0.113.7", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            rs = await asyncio.gather(c.post("/api/scenarios"), c.post("/api/scenarios"))
            return sorted(r.status_code for r in rs)

    t0 = time.monotonic()
    assert asyncio.run(run()) == [200, 503]
    assert time.monotonic() - t0 < 5


def test_limits_disabled_by_zero() -> None:
    app = _app(per_min=0, max_heavy=0)
    c = TestClient(app, client=("203.0.113.8", 1))
    assert all(c.get("/api/matches").status_code == 200 for _ in range(50))


def test_security_headers_and_head_on_health() -> None:
    from footymind.api.app import app

    c = TestClient(app)
    r = c.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert c.head("/api/health").status_code == 200  # uptime monitors often use HEAD


def test_unknown_extensionless_paths_open_the_lab_but_missing_assets_stay_404(
    tmp_path: Path,
) -> None:
    from footymind.api.app import SpaStaticFiles

    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>")  # SYNTHETIC
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    a = FastAPI()
    a.mount("/", SpaStaticFiles(directory=tmp_path, html=True), name="web")
    c = TestClient(a)
    assert c.get("/").status_code == 200
    r = c.get("/about")
    assert r.status_code == 200 and "id=root" in r.text
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/assets/old-build.js").status_code == 404
