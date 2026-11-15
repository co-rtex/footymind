"""API robustness and contract tests on real data (skipped when the data is absent).

Every failure mode must fail clearly (4xx/503 with a reason) and never fabricate a value:
unknown ids, invalid numbers, non-finite inputs, missing players, frames without a ball or a
possession, invalid response-policy parameters, missing or corrupt similarity artefacts, and a
provider without an option model (which must never borrow another provider's model).
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from footymind.data.store import available_matches, load_match
from footymind.research import similarity
from footymind.spatial import option_model
from footymind.spatial.state import attacking_team_at

MATCHES = available_matches()
pytestmark = pytest.mark.skipif(
    not {"metrica-sample-1", "metrica-sample-2"} <= set(MATCHES),
    reason="processed Metrica data not available",
)
MID = "metrica-sample-1"
FEATURED = 53708  # the lab's featured frame: away in possession, home_9 on the pitch
SC = next((m for m in ("skillcorner-2017461", "skillcorner-1899585") if m in MATCHES), None)
needs_sc = pytest.mark.skipif(SC is None, reason="processed SkillCorner data not available")


def _strict(text: str) -> Any:
    def bad(name: str) -> Any:
        raise AssertionError(f"non-finite JSON constant {name}")

    return json.loads(text, parse_constant=bad)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from footymind.api.app import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def index_rows() -> pl.DataFrame:
    """A small real index (sample 2 only) injected in place of the on-disk index."""
    from footymind.research.pass_dataset import build

    return build(["metrica-sample-2"], similarity.similarity_dataset_config()).select(
        similarity.STORED_COLUMNS
    )


@pytest.fixture
def metrica_index(index_rows: pl.DataFrame):  # type: ignore[no-untyped-def]
    old = similarity._CACHE.pop("metrica", None)
    similarity._CACHE["metrica"] = similarity.SimilarityIndex.from_rows(index_rows)
    yield
    similarity._CACHE.pop("metrica", None)
    if old is not None:
        similarity._CACHE["metrica"] = old


def _no_ball_frame(match_id: str) -> int:
    """A frame inside a possession where the ball is unobserved (Metrica: out of play)."""
    b = load_match(match_id)
    td = b.tracking
    for r in np.flatnonzero(np.isnan(td.ball_xy).any(axis=1))[::500]:
        f = int(td.frame[r])
        if attacking_team_at(b, f)[0] is not None:
            return f
    pytest.skip("no frame without a ball inside a possession")


def _scenario(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "match_id": MID,
        "frame": FEATURED,
        "interventions": [{"player_id": "home_9", "x": 50.7, "y": -8.75}],
        "ensemble": 4,
    }
    body.update(kw)
    return body


# ---------- unknown ids and invalid numbers ----------


@pytest.mark.parametrize(
    "path",
    [
        "/api/matches/no-such-match",
        "/api/matches/no-such-match/possessions",
        "/api/matches/no-such-match/replay?start_frame=1&end_frame=10",
        "/api/matches/no-such-match/frames/1/analysis",
        "/api/matches/no-such-match/frames/1/similar",
        "/api/matches/no-such-match/frames/1/possession",
        "/api/matches/no-such-match/frames/1/defensive-shadow",
        "/api/response-policies?match_id=no-such-match",
        "/api/possessions/no-such-match:p0001/replay",
        "/api/possessions/garbage/replay",
        f"/api/possessions/{MID}:p9999/replay",
        f"/api/matches/{MID}/frames/-5/analysis",
        f"/api/matches/{MID}/frames/{10**30}/analysis",
        f"/api/matches/{MID}/frames/{10**30}/similar",
        f"/api/matches/{MID}/replay?start_frame=-10&end_frame=-1",
    ],
)
def test_unknown_ids_are_404(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 404, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [
        f"/api/matches/{MID}/frames/abc/analysis",
        f"/api/matches/{MID}/frames/1.5/analysis",
        f"/api/matches/{MID}/replay?start_frame=10&end_frame=1",
        f"/api/matches/{MID}/replay?start_frame=1&end_frame={10**30}",  # window too large
        f"/api/matches/{MID}/replay?start_frame=1&end_frame=100&step=0",
        f"/api/matches/{MID}/possessions?min_passes=-1",
        f"/api/matches/{MID}/possessions?team=referee",
        f"/api/possessions/{MID}:p0001/replay?pad_s=nan",
        f"/api/matches/{MID}/frames/{FEATURED}/similar?k=0",
        f"/api/matches/{MID}/frames/{FEATURED}/similar?k=51",
        f"/api/matches/{MID}/frames/{FEATURED}/defensive-shadow?horizon_s=0",
        f"/api/matches/{MID}/frames/{FEATURED}/defensive-shadow?horizon_s=inf",
        f"/api/matches/{MID}/frames/{FEATURED}/defensive-shadow?value_model=oracle",
    ],
)
def test_invalid_parameters_are_422(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 422, r.text
    _strict(r.text)


# ---------- scenarios: coordinates, players, policies and their parameters ----------


@pytest.mark.parametrize(
    "change",
    [
        {"interventions": []},
        {"interventions": [{"player_id": "home_9", "x": 50.7, "y": -8.75}] * 5},
        {"interventions": [{"player_id": "home_9", "x": 61.0, "y": 0.0}]},
        {"interventions": [{"player_id": "home_9", "x": 0.0, "y": -40.5}]},
        {"interventions": [{"player_id": "home_9", "x": "left", "y": 0.0}]},
        # one edit per player: two different targets for the same player are ambiguous
        {
            "interventions": [
                {"player_id": "home_9", "x": 1.0, "y": 0.0},
                {"player_id": "home_9", "x": 5.0, "y": 5.0},
            ]
        },
        {"response_policy": "bogus"},
        {"response_policy": None},
        {"horizon_s": 0.0},
        {"horizon_s": -1.0},
        {"horizon_s": 5.01},
        {"follow_fraction": -0.01},
        {"follow_fraction": 1.01},
        {"ensemble": -1},
        {"ensemble": 65},
        {"velocity_mode": "teleport"},
        {"value_model": "oracle"},
        {"frame": "abc"},
    ],
)
def test_invalid_scenario_requests_are_422(client: TestClient, change: dict[str, Any]) -> None:
    r = client.post("/api/scenarios", json=_scenario(**change))
    assert r.status_code == 422, r.text
    _strict(r.text)


@pytest.mark.parametrize("field", ["x", "horizon_s", "follow_fraction"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_inputs_are_422_with_valid_json(
    client: TestClient, field: str, token: str
) -> None:
    # Python's JSON parser accepts these tokens. Before the fix the 422 echoed the NaN input
    # and could not be serialised, so the client got a 500 instead of a reason.
    body = json.dumps(_scenario()).replace('"x": 50.7', '"x": 1.0')
    if field == "x":
        body = body.replace('"x": 1.0', f'"x": {token}')
    else:
        body = body.replace('"ensemble": 4', f'"ensemble": 4, "{field}": {token}')
    r = client.post("/api/scenarios", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    d = _strict(r.text)["detail"]
    assert any(e["type"] == "finite_number" for e in d), d


def test_scenario_unknown_match_frame_and_player(client: TestClient) -> None:
    assert client.post("/api/scenarios", json=_scenario(match_id="nope")).status_code == 404
    assert client.post("/api/scenarios", json=_scenario(frame=10**12)).status_code == 404
    r = client.post(
        "/api/scenarios",
        json=_scenario(interventions=[{"player_id": "nobody", "x": 0.0, "y": 0.0}]),
    )
    assert r.status_code == 404 and "nobody" in r.json()["detail"]


def test_scenario_on_absent_player_is_422(client: TestClient) -> None:
    td = load_match(MID).tracking
    row = td.index_of_frame(FEATURED)
    absent = [p.player_id for i, p in enumerate(td.players) if np.isnan(td.xy[row, i]).any()]
    assert absent, "expected substitutes off the pitch"
    r = client.post(
        "/api/scenarios", json=_scenario(interventions=[{"player_id": absent[0], "x": 0, "y": 0}])
    )
    assert r.status_code == 422 and "not on the pitch" in r.json()["detail"]


@pytest.mark.parametrize("ensemble", [0, 1])
def test_scenario_parameter_bounds_accepted(client: TestClient, ensemble: int) -> None:
    r = client.post(
        "/api/scenarios",
        json=_scenario(
            ensemble=ensemble, follow_fraction=1.0, horizon_s=5.0, response_policy="lane_cover"
        ),
    )
    assert r.status_code == 200, r.text
    body = _strict(r.text)
    assert body["policy"]["name"] == "lane_cover"
    assert body["ensemble_size"] == ensemble
    if ensemble == 0:  # no ensemble: no robustness claim
        assert all(d["delta_sign_agreement"] is None for d in body["option_deltas"])


def test_learned_policy_states_the_forecast_horizon_it_used(client: TestClient) -> None:
    # The learned model is trained at fixed offsets; a 5 s request uses the longest one and says so.
    r = client.post("/api/scenarios", json=_scenario(response_policy="learned", horizon_s=5.0))
    assert r.status_code == 200, r.text
    sim = [m for m in r.json()["movements"] if m["provenance"] == "simulated"]
    assert all(re.search(r"\(\d+\.\d\d s forecast\)", m["reason"]) for m in sim)


def test_learned_policy_without_a_model_fails_clearly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from footymind.counterfactual import scenario as sc

    monkeypatch.setattr(sc, "load_response_model", lambda _provider: None)
    r = client.post("/api/scenarios", json=_scenario(response_policy="learned"))
    assert r.status_code == 422 and "not available" in r.json()["detail"]


# ---------- frames with no ball or no possession ----------


def test_frame_without_ball(client: TestClient) -> None:
    f = _no_ball_frame(MID)
    a = _strict(client.get(f"/api/matches/{MID}/frames/{f}/analysis").text)
    assert a["ball_x"] is None and a["ball_y"] is None
    assert a["pass_options"] == [] and a["pressure"] is None and a["carrier_id"] is None
    assert any("ball position not observed" in w for w in a["warnings"])
    r = client.get(f"/api/matches/{MID}/frames/{f}/defensive-shadow")
    assert r.status_code == 422 and "ball" in r.json()["detail"]
    s = client.post(
        "/api/scenarios",
        json=_scenario(frame=f, interventions=[{"player_id": "home_9", "x": 10.0, "y": 0.0}]),
    )
    if s.status_code == 422:  # home_9 may be off the pitch at that frame
        return
    body = _strict(s.text)
    assert body["option_deltas"] == []
    # No options can be counted without a ball: unknown, never 0 (fixed in the API).
    for summ in (body["summary_observed"], body["summary_hypothetical"]):
        assert summ["n_options_physics_over_0_5"] is None
        assert summ["best_option_receiver"] is None


@needs_sc
def test_frame_without_possession(client: TestClient) -> None:
    assert SC is not None
    b = load_match(SC)
    f = int(b.tracking.frame[0])
    assert attacking_team_at(b, f)[0] is None
    a = _strict(client.get(f"/api/matches/{SC}/frames/{f}/analysis").text)
    assert a["attacking_team"] is None and a["pass_options"] == [] and a["temporal"] is None
    assert a["warnings"] == ["no possession"]
    assert client.get(f"/api/matches/{SC}/frames/{f}/possession").status_code == 404
    pid = next(p.player_id for p in b.tracking.players)
    r = client.post(
        "/api/scenarios",
        json={"match_id": SC, "frame": f, "interventions": [{"player_id": pid, "x": 0, "y": 0}]},
    )
    assert r.status_code == 422 and "no possession" in r.json()["detail"]


# ---------- similar moments: index and validation artefacts ----------


def test_missing_similarity_index_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(similarity, "_CACHE", {})
    monkeypatch.setattr(similarity, "index_path", lambda p: tmp_path / f"similarity_{p}.parquet")
    r = client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar")
    assert r.status_code == 503 and "build-similarity-index" in r.json()["detail"]


@pytest.mark.parametrize("content", [b"not a parquet file", b""])
def test_corrupt_similarity_index_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes
) -> None:
    bad = tmp_path / "similarity_metrica.parquet"
    bad.write_bytes(content)
    monkeypatch.setattr(similarity, "_CACHE", {})
    monkeypatch.setattr(similarity, "index_path", lambda _p: bad)
    r = client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar")
    assert r.status_code == 503, r.text
    assert "unreadable" in r.json()["detail"] and "rebuild" in r.json()["detail"]
    assert "metrica" not in similarity._CACHE  # a later rebuild is picked up


def test_index_with_wrong_columns_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "similarity_metrica.parquet"
    pl.DataFrame({"match_id": ["metrica-sample-2"], "frame": [1]}).write_parquet(bad)
    monkeypatch.setattr(similarity, "_CACHE", {})
    monkeypatch.setattr(similarity, "index_path", lambda _p: bad)
    assert client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar").status_code == 503


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"metrics": 3}'])
def test_corrupt_validation_results_is_503(
    client: TestClient,
    metrica_index: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: str,
) -> None:
    from footymind.api import app as app_mod

    d = tmp_path / "results" / "similarity_validation_v1"
    d.mkdir(parents=True)
    (d / "results.json").write_text(content)
    monkeypatch.setattr(app_mod, "EXPERIMENTS_DIR", tmp_path)
    r = client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar")
    assert r.status_code == 503 and "unreadable" in r.json()["detail"]


def test_missing_validation_results_is_null(
    client: TestClient, metrica_index: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from footymind.api import app as app_mod

    monkeypatch.setattr(app_mod, "EXPERIMENTS_DIR", tmp_path)
    body = _strict(client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar").text)
    assert body["validation"] is None and body["estimate"] is not None


def test_no_neighbours_gives_null_share_not_zero(
    client: TestClient, index_rows: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An index holding only the queried match (e.g. a one-match deployment): every row is
    # excluded, so there is nothing to compare with.
    own = index_rows.with_columns(pl.lit(MID).alias("match_id"))
    monkeypatch.setattr(
        similarity, "_CACHE", {"metrica": similarity.SimilarityIndex.from_rows(own)}
    )
    body = _strict(client.get(f"/api/matches/{MID}/frames/{FEATURED}/similar").text)
    assert body["k"] == 0 and body["neighbours"] == []
    assert body["completed_share"] is None and body["estimate"] is None


# ---------- per-provider option models: never borrow another provider's model ----------


def test_option_model_resources_are_per_provider() -> None:
    res = option_model.PROVIDER_RESOURCES
    assert len(set(res.values())) == len(res), "two providers share a model file"
    for provider in res:
        m = option_model.load_default(f"{provider}-x")
        if m is not None:  # every packaged model names its provider (a null one is refused)
            assert m.meta.get("provider") == provider
    assert option_model.load_default("statsbomb-1") is None  # unknown provider: no model


@needs_sc
def test_served_option_model_names_its_provider(client: TestClient) -> None:
    assert SC is not None
    for mid in (MID, SC):
        f = int(load_match(mid).tracking.frame[0])
        a = _strict(client.get(f"/api/matches/{mid}/frames/{f}/analysis?grid=false").text)
        if a["option_model"] is not None:
            assert a["option_model"]["provider"] == mid.split("-")[0]
            assert (
                a["option_model"]["resource"] == option_model.PROVIDER_RESOURCES[mid.split("-")[0]]
            )


@needs_sc
def test_provider_without_option_model_falls_back_to_physics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert SC is not None
    res = {k: v for k, v in option_model.PROVIDER_RESOURCES.items() if k != "skillcorner"}
    monkeypatch.setattr(option_model, "PROVIDER_RESOURCES", res)
    b = load_match(SC)
    passes = b.events.filter(pl.col("type") == "PASS")
    f = int(passes["start_frame"][len(passes) // 2])
    a = _strict(client.get(f"/api/matches/{SC}/frames/{f}/analysis?grid=false").text)
    assert a["option_model"] is None
    assert a["pass_options"], "physics options must still be served"
    assert all(o["p_success_model"] is None for o in a["pass_options"])  # never Metrica's model
    att = a["attacking_team"]
    mover = next(
        p for p in a["players"] if p["team"] == att and p["x"] is not None and not p["is_gk"]
    )
    r = client.post(
        "/api/scenarios",
        json={
            "match_id": SC,
            "frame": f,
            "interventions": [{"player_id": mover["player_id"], "x": mover["x"], "y": mover["y"]}],
            "ensemble": 2,
        },
    )
    assert r.status_code == 200, r.text
    body = _strict(r.text)
    assert body["value_model"] == "physics"
    assert body["observed"]["option_model"] is None
    # Metrica still gets its own model.
    m = _strict(client.get(f"/api/matches/{MID}/frames/{FEATURED}/analysis?grid=false").text)
    assert m["option_model"] is not None and m["option_model"]["provider"] == "metrica"


@needs_sc
def test_model_labelled_for_another_provider_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert SC is not None
    real = option_model.load_default(MID)
    assert real is not None
    wrong = dataclasses.replace(real, meta={**real.meta, "provider": "metrica"})
    cache = dict(option_model._CACHE)
    cache[option_model.PROVIDER_RESOURCES["skillcorner"]] = wrong
    monkeypatch.setattr(option_model, "_CACHE", cache)
    f = int(load_match(SC).tracking.frame[0])
    r = client.get(f"/api/matches/{SC}/frames/{f}/analysis?grid=false")
    assert r.status_code == 503 and "refused" in r.json()["detail"]
    pid = load_match(SC).tracking.players[0].player_id
    body = {"match_id": SC, "frame": f, "interventions": [{"player_id": pid, "x": 0, "y": 0}]}
    r = client.post("/api/scenarios", json=body)
    assert r.status_code == 503 and "refused" in r.json()["detail"]


# ---------- replay contract: the ball detection mask ----------


def test_replay_ball_detected_is_null_on_metrica(client: TestClient) -> None:
    r = _strict(client.get(f"/api/matches/{MID}/replay?start_frame=100&end_frame=200").text)
    assert r["ball_detected"] is None


@needs_sc
@pytest.mark.parametrize("step", [1, 3])
def test_replay_ball_detected_on_skillcorner(client: TestClient, step: int) -> None:
    assert SC is not None
    td = load_match(SC).tracking
    if td.ball_detected is None:
        pytest.skip("bundle ingested before ball_detected existed")
    pos = client.get(f"/api/matches/{SC}/possessions?min_passes=3").json()
    r = _strict(client.get(f"/api/possessions/{pos[0]['possession_id']}/replay?step={step}").text)
    n = len(r["frame"])
    bd = r["ball_detected"]
    assert isinstance(bd, list) and len(bd) == n and all(isinstance(v, bool) for v in bd)
    rows = [td.index_of_frame(f) for f in r["frame"]]
    for k, (row, det) in enumerate(zip(rows, bd, strict=True)):
        null = r["ball_xy"][2 * k] is None
        assert det == (bool(td.ball_detected[row]) and not null)
    assert any(not v for v in bd) or bool(td.ball_detected[rows].all())


# ---------- numerical stability: no NaN/Infinity in any response ----------


def _sample_frames(match_id: str, n: int, seed: int = 0) -> list[int]:
    ev = load_match(match_id).events.filter(pl.col("type") == "PASS")
    rng = np.random.default_rng(seed)
    idx = rng.choice(ev.height, size=min(n, ev.height), replace=False)
    return [int(ev["start_frame"][int(i)]) for i in idx]


def test_responses_never_contain_non_finite_numbers(
    client: TestClient, metrica_index: None
) -> None:
    mids = [MID] + ([SC] if SC else [])
    n_checked = 0
    for mid in mids:
        frames = _sample_frames(mid, 12)
        frames += (
            [int(load_match(mid).tracking.frame[0]), _no_ball_frame(MID)] if mid == MID else []
        )
        for f in frames:
            for path in (
                f"/api/matches/{mid}/frames/{f}/analysis",
                f"/api/matches/{mid}/frames/{f}/defensive-shadow",
                f"/api/matches/{mid}/replay?start_frame={f - 50}&end_frame={f + 50}",
            ):
                r = client.get(path)
                assert r.status_code < 500, (path, r.text[:300])
                _strict(r.text)
                n_checked += 1
            if mid == MID:
                r = client.get(f"/api/matches/{mid}/frames/{f}/similar")
                assert r.status_code < 500, r.text[:300]
                _strict(r.text)
        for f in frames[:3]:
            a = client.get(f"/api/matches/{mid}/frames/{f}/analysis?grid=false").json()
            mover = next(
                (
                    p
                    for p in a["players"]
                    if p["team"] == a["attacking_team"] and p["x"] is not None
                ),
                None,
            )
            if mover is None:
                continue
            for policy in ("none", "marker_follows", "zonal_shift", "learned"):
                r = client.post(
                    "/api/scenarios",
                    json={
                        "match_id": mid,
                        "frame": f,
                        "interventions": [
                            {
                                "player_id": mover["player_id"],
                                "x": max(-55.0, min(55.0, mover["x"] + 4.0)),
                                "y": mover["y"],
                            }
                        ],
                        "response_policy": policy,
                        "ensemble": 4,
                    },
                )
                assert r.status_code < 500, (policy, r.text[:300])
                _strict(r.text)
                n_checked += 1
    assert n_checked >= 40


# ---------- caches under concurrency ----------


def test_concurrent_requests_match_sequential_results() -> None:
    """Threaded calls across more matches than the bundle LRU holds (4) agree with serial ones."""
    from footymind.api.app import frame_analysis, replay

    mids = [m for m in MATCHES if m.startswith("metrica-")] + [
        m for m in MATCHES if m.startswith("skillcorner-")
    ][:3]
    jobs = [(m, f) for m in mids for f in _sample_frames(m, 2, seed=1)]

    def run(job: tuple[str, int]) -> tuple[str, str]:
        m, f = job
        a = frame_analysis(m, f, grid=False).model_dump_json()
        r = replay(m, f - 25, f + 25, 5, None).model_dump_json()
        return a, r

    serial = [run(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=6) as ex:
        threaded = list(ex.map(run, jobs * 2))
    assert threaded == serial * 2


def test_lru_bundle_cache_is_consistent_after_eviction() -> None:
    from footymind.data.store import load_match as lm

    a = lm(MID).tracking.xy[1000].copy()
    for m in MATCHES[:6]:
        lm(m)
    assert np.array_equal(lm(MID).tracking.xy[1000], a, equal_nan=True)


def test_f_rounding_never_emits_non_finite() -> None:
    from footymind.api.app import _f, _flat, _json_safe

    assert _f(float("nan")) is None and _f(float("inf")) is None and _f(None) is None
    assert _f(-0.0001) == 0.0 and math.copysign(1.0, _f(-0.0001) or 0.0) == 1.0
    assert _flat(np.array([[np.nan, 1.234, np.inf]])) == [None, 1.23, None]
    assert _json_safe({"a": [float("nan"), 1.0, {"b": float("-inf")}]}) == {
        "a": ["nan", 1.0, {"b": "-inf"}]
    }


def test_response_models_labelled_for_another_provider_are_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C69 for the response models (data-free): a packaged file labelled for Metrica, mapped to
    SkillCorner by mistake, is treated as absent instead of being served."""
    from footymind.counterfactual import response_innovation, response_model

    for mod, resources_name, load in (
        (response_model, "PROVIDER_RESOURCES", response_model.load_response_model),
        (response_innovation, "INNOVATION_RESOURCES", response_innovation.load_innovation_model),
    ):
        res = dict(getattr(mod, resources_name))
        res["skillcorner"] = res["metrica"]  # a Metrica-labelled file under the SkillCorner key
        monkeypatch.setattr(mod, resources_name, res)
        monkeypatch.setattr(mod, "_CACHE", {})
        with pytest.warns(UserWarning, match="labelled for provider 'metrica'"):
            assert load("skillcorner") is None
        monkeypatch.setattr(mod, "_CACHE", {})
        m = load("metrica")
        assert m is not None and m.meta["provider"] == "metrica"


def test_similar_validation_applies_only_to_the_validated_index() -> None:
    """Audit 2026-11-15: the public demo serves a 3-match SkillCorner index while the held-out
    validation (C36) used all 20 matches. The summary must say it does not apply there."""
    import json

    from footymind.api.app import _similarity_validation
    from footymind.paths import EXPERIMENTS_DIR

    res = EXPERIMENTS_DIR / "results" / "similarity_validation_v1_skillcorner" / "results.json"
    validated = set(json.loads(res.read_text())["matches"])
    assert len(validated) == 20
    demo = {"skillcorner-1874553", "skillcorner-1899585", "skillcorner-2017461"}
    v = _similarity_validation("skillcorner", demo)
    assert v is not None and v["applies_to_this_index"] is False
    assert (v["validated_matches"], v["index_matches"]) == (20, 3)
    assert v["knn_beats_base_rate"] is True  # the result itself is reported unchanged
    full = _similarity_validation("skillcorner", validated)
    assert full is not None and full["applies_to_this_index"] is True
    unknown = _similarity_validation("skillcorner")
    assert unknown is not None and unknown["applies_to_this_index"] is None
