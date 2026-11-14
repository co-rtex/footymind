"""Request limits, error bodies and option-model packaging (no match data needed)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from footymind.api.app import MAX_BODY_BYTES, app
from footymind.spatial import option_model

client = TestClient(app, raise_server_exceptions=False)


def test_validation_error_does_not_echo_the_input() -> None:
    secret = "x" * 5000
    r = client.post("/api/scenarios", json={"match_id": 3, "unexpected": secret})
    assert r.status_code == 422
    errors = r.json()["detail"]
    assert errors and all("input" not in e for e in errors)
    assert all({"loc", "msg", "type"} <= set(e) for e in errors)
    assert secret not in r.text


def test_oversized_body_is_413() -> None:
    body = json.dumps({"match_id": "metrica-sample-1", "pad": "x" * (MAX_BODY_BYTES + 1)})
    r = client.post("/api/scenarios", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]


def test_oversized_chunked_body_is_413() -> None:
    def chunks() -> Iterator[bytes]:  # no Content-Length: the size is counted while reading
        yield b'{"pad": "'
        for _ in range(MAX_BODY_BYTES // 65536 + 2):
            yield b"x" * 65536
        yield b'"}'

    r = client.post(
        "/api/scenarios", content=chunks(), headers={"content-type": "application/json"}
    )
    assert r.status_code == 413 and "larger than" in r.json()["detail"]


def test_small_body_is_not_refused() -> None:
    r = client.post("/api/scenarios", json={})
    assert r.status_code == 422  # reaches validation: not 413


def test_packaged_option_models_name_their_provider() -> None:
    for provider, resource in option_model.PROVIDER_RESOURCES.items():
        m = option_model.load_default(f"{provider}-x")
        assert m is not None, resource
        assert m.meta.get("provider") == provider, resource


@pytest.mark.parametrize("label", [None, "skillcorner"])
def test_option_model_without_or_with_another_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch, label: str | None
) -> None:
    from footymind.api.app import _provider_model

    real = option_model.load_default("metrica-x")
    assert real is not None
    meta = {k: v for k, v in real.meta.items() if k != "provider"}
    if label is not None:
        meta["provider"] = label
    cache = dict(option_model._CACHE)
    cache[option_model.PROVIDER_RESOURCES["metrica"]] = dataclasses.replace(real, meta=meta)
    monkeypatch.setattr(option_model, "_CACHE", cache)
    with pytest.raises(HTTPException) as e:
        _provider_model("metrica-sample-1")
    assert e.value.status_code == 503 and "refused" in str(e.value.detail)
