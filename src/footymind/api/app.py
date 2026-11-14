"""FootyMind HTTP API (FastAPI). Run with ``footymind serve``."""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from footymind import __version__
from footymind.api import schemas as S
from footymind.counterfactual.response import POLICY_NAMES, PolicyInfo, policy_info
from footymind.counterfactual.scenario import REACH_SPEED_MPS, Intervention, run_scenario
from footymind.data.store import (
    PIPELINE_VERSION,
    MatchBundle,
    StaleBundleError,
    available_matches,
    load_match,
    read_manifest,
)
from footymind.ghost.defensive_shadow import NOISE_FLOOR as SHADOW_NOISE_FLOOR
from footymind.ghost.defensive_shadow import defensive_shadow
from footymind.paths import EXPERIMENTS_DIR, REPO_ROOT
from footymind.spatial.analysis import FrameAnalysis, analyze
from footymind.spatial.explain import explain
from footymind.spatial.option_model import (
    PROVIDER_RESOURCES,
    OptionModel,
    load_default,
    provider_of,
)
from footymind.spatial.pitch_control import DEFAULT_GRID, DEFAULT_PARAMS
from footymind.spatial.state import attacking_team_at, frame_state
from footymind.spatial.temporal import temporal_context
from footymind.tracking.model import TrackingData

MAX_REPLAY_FRAMES = 4000

app = FastAPI(
    title="FootyMind API",
    version=__version__,
    description=(
        "Tracking-based football analysis: observed replay, spatial analytics and "
        "constrained counterfactual scenarios. All analytical outputs carry provenance labels."
    ),
)
MAX_BODY_BYTES = 1_000_000  # a scenario request is a few kB; larger bodies are refused


class _BodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(413, f"request body larger than {MAX_BODY_BYTES} bytes")


class BodySizeLimit:
    """Refuse request bodies above ``max_bytes`` with a 413 (pure ASGI, so streaming works).

    A declared ``Content-Length`` is checked before anything is read. A body without one
    (chunked) is counted while it is read; FastAPI re-raises the HTTPException from the body
    reader, so the client gets the same 413.
    """

    def __init__(self, app: Any, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        too_large = JSONResponse(
            status_code=413, content={"detail": f"request body larger than {self.max_bytes} bytes"}
        )
        try:
            if declared is not None and int(declared) > self.max_bytes:
                await too_large(scope, receive, send)
                return
        except ValueError:
            await JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})(
                scope, receive, send
            )
            return
        seen = 0

        async def counted() -> Any:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        await self.app(scope, counted, send)


app.add_middleware(BodySizeLimit)
app.add_middleware(GZipMiddleware, minimum_size=2048)
_origins = os.environ.get("FOOTYMIND_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in _origins.split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_header(request: Request, call_next: Any) -> Response:
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with their names so an error body is valid JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    return x


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # The default handler echoes the rejected input. A NaN or Infinity in the request body
    # (accepted by Python's JSON parser) then made the 422 itself unserialisable: a 500, and
    # echoing a whole (possibly large) body back is needless. Errors keep their location,
    # message and type; the input is dropped.
    errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    detail = _json_safe(jsonable_encoder(errors))
    return JSONResponse(status_code=422, content={"detail": detail})


def _f(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd) + 0.0  # no -0.0


def _bundle(match_id: str) -> MatchBundle:
    if match_id not in available_matches():
        raise HTTPException(404, f"unknown match {match_id!r}")
    try:
        return load_match(match_id)
    except StaleBundleError as e:
        raise HTTPException(503, str(e)) from e


def _source(b: MatchBundle) -> S.SourceInfo:
    p = b.manifest["provenance"]
    return S.SourceInfo(
        provider=p["provider"],
        name=p["source_name"],
        url=p["source_url"],
        license_note=p["license_note"],
    )


def _possession_out(r: dict[str, Any], fps: float) -> S.PossessionSummary:
    return S.PossessionSummary(
        possession_id=r["possession_id"],
        team=r["team"],
        period=r["period"],
        start_frame=r["start_frame"],
        end_frame=r["end_frame"],
        start_time_s=r["start_time_s"],
        duration_s=round((r["end_frame"] - r["start_frame"]) / fps, 2),
        n_events=r["n_events"],
        n_passes=r["n_passes"],
        first_type=r["first_type"],
        first_subtype=r["first_subtype"],
        last_type=r["last_type"],
        last_subtype=r["last_subtype"],
        has_shot=r["has_shot"],
        has_goal=r["has_goal"],
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "matches": available_matches()}


@app.get("/api/matches", response_model=list[S.MatchSummary])
def list_matches() -> list[S.MatchSummary]:
    return [match_summary(m) for m in available_matches()]


@app.get("/api/matches/{match_id}", response_model=S.MatchSummary)
def match_summary(match_id: str) -> S.MatchSummary:
    # Built from the small manifest only: loading every tracking bundle here made
    # /api/matches take ~2.5 s per call with 23 matches (bundle LRU cache thrashing).
    if match_id not in available_matches():
        raise HTTPException(404, f"unknown match {match_id!r}")
    m = _manifest(match_id)
    q = m["quality"]["tracking"]
    periods = [
        S.PeriodInfo(
            period=int(k),
            **{
                kk: v[kk]
                for kk in ("first_frame", "last_frame", "duration_s", "home_attacks_positive_x")
            },
        )
        for k, v in sorted(q["periods"].items())
    ]
    p = m["provenance"]
    return S.MatchSummary(
        match_id=match_id,
        source=S.SourceInfo(
            provider=p["provider"],
            name=p["source_name"],
            url=p["source_url"],
            license_note=p["license_note"],
        ),
        fps=float(m["fps"]),
        n_frames=int(q["n_frames"]),
        periods=periods,
        n_possessions=int(m["counts"]["possessions"]),
        n_events=int(m["counts"]["events"]),
        pitch=S.Pitch(length=float(m["pitch"]["length"]), width=float(m["pitch"]["width"])),
        **_match_meta(match_id),
    )


def _match_meta(match_id: str) -> dict[str, str]:
    """Public match facts from the provider's (checksum-verified) metadata file.

    Only team short names, the competition, the round and the date are read: no player
    names. Metrica's sample data is anonymised, so nothing is returned for it, and nothing
    is returned when the raw file is not present (for example, processed-only deployments).
    """
    from footymind.data.sources import SOURCES

    src = SOURCES.get(match_id)
    if src is None or src.fmt != "skillcorner":
        return {}
    num = match_id.split("-", 1)[1]
    path = src.local_dir() / f"{num}_match.json"
    try:
        st = os.stat(path)
    except OSError:
        return {}
    # cached per file version: a replaced metadata file is read again on the next call
    return _read_match_meta(path, (st.st_mtime_ns, st.st_size, st.st_ino))


@lru_cache(maxsize=256)
def _read_match_meta(path: Path, _token: tuple[int, int, int]) -> dict[str, str]:
    try:
        d = json.loads(path.read_text())
        out = {
            "home_label": str(d["home_team"]["short_name"]),
            "away_label": str(d["away_team"]["short_name"]),
        }
        comp = d.get("competition_edition", {}).get("competition", {}).get("name")
        rnd = (d.get("competition_round") or {}).get("name")
        if comp:
            out["competition"] = f"{comp}, {rnd}" if rnd else str(comp)
        if d.get("date_time"):
            out["date"] = str(d["date_time"])[:10]
        return out
    except (KeyError, TypeError, ValueError):
        return {}


def _manifest(match_id: str) -> dict[str, Any]:
    # read_manifest is cached per manifest file version (one os.stat per call), so a
    # re-ingest while the server runs is served on the next request.
    m = read_manifest(match_id)
    if m["pipeline_version"] != PIPELINE_VERSION:
        raise HTTPException(503, f"{match_id}: stale bundle; re-run `footymind ingest {match_id}`")
    return m


@app.get("/api/matches/{match_id}/possessions", response_model=list[S.PossessionSummary])
def list_possessions(
    match_id: str,
    min_passes: int = Query(0, ge=0),
    team: S.Team | None = None,
) -> list[S.PossessionSummary]:
    b = _bundle(match_id)
    pos = b.possessions.filter(pl.col("n_passes") >= min_passes)
    if team:
        pos = pos.filter(pl.col("team") == team)
    return [_possession_out(r, b.tracking.fps) for r in pos.iter_rows(named=True)]


def _events_out(ev: pl.DataFrame) -> list[S.EventOut]:
    out = []
    for r in ev.iter_rows(named=True):
        out.append(
            S.EventOut(
                **{
                    k: r[k]
                    for k in (
                        "event_id",
                        "team",
                        "type",
                        "subtype",
                        "period",
                        "start_frame",
                        "end_frame",
                        "start_time_s",
                        "from_player_id",
                        "to_player_id",
                        "possession_id",
                    )
                },
                start_x=_f(r["start_x"], 2),
                start_y=_f(r["start_y"], 2),
                end_x=_f(r["end_x"], 2),
                end_y=_f(r["end_y"], 2),
            )
        )
    return out


def _period_time(td: TrackingData, rows: np.ndarray) -> np.ndarray:
    """Seconds since each frame's period started (first tracked frame of that period).

    Provider clocks differ: Metrica time is cumulative across periods, and SkillCorner time
    is video time that does not start at kick-off. This gives clients one basis.
    """
    starts = {int(p): float(td.time_s[np.argmax(td.period == p)]) for p in np.unique(td.period)}
    return np.array([td.time_s[r] - starts[int(td.period[r])] for r in rows], dtype=np.float64)


def _flat(a: np.ndarray, nd: int = 2) -> list[float | None]:
    r = np.round(a.astype(np.float64).ravel(), nd)
    return [float(v) if math.isfinite(v) else None for v in r.tolist()]


@app.get("/api/matches/{match_id}/replay", response_model=S.Replay)
def replay(
    match_id: str,
    start_frame: int,
    end_frame: int,
    step: int = Query(1, ge=1, le=25),
    possession_id: str | None = None,
) -> S.Replay:
    b = _bundle(match_id)
    td = b.tracking
    if end_frame < start_frame:
        raise HTTPException(422, "end_frame < start_frame")
    lo = int(np.searchsorted(td.frame, start_frame, "left"))
    hi = int(np.searchsorted(td.frame, end_frame, "right"))
    rows = np.arange(lo, hi, step)
    if rows.size == 0:
        raise HTTPException(404, "no frames in range")
    if rows.size > MAX_REPLAY_FRAMES:
        raise HTTPException(422, f"window too large ({rows.size} > {MAX_REPLAY_FRAMES} frames)")
    f0, f1 = int(td.frame[rows[0]]), int(td.frame[rows[-1]])
    ev = b.events.filter((pl.col("start_frame") >= f0) & (pl.col("start_frame") <= f1))
    poss = None
    if possession_id:
        hit = b.possessions.filter(pl.col("possession_id") == possession_id)
        if hit.height:
            poss = _possession_out(hit.row(0, named=True), td.fps)
    signs = {
        str(p): {"home": td.attack_sign("home", p), "away": td.attack_sign("away", p)}
        for p in sorted(td.home_attacks_positive_x)
    }
    return S.Replay(
        match_id=match_id,
        fps=td.fps,
        step=step,
        pitch=S.Pitch(length=td.pitch.length, width=td.pitch.width),
        players=[
            S.PlayerOut(player_id=p.player_id, team=p.team, jersey=p.jersey) for p in td.players
        ],
        frame=td.frame[rows].tolist(),
        time_s=np.round(td.time_s[rows], 2).tolist(),
        period_time_s=np.round(_period_time(td, rows), 2).tolist(),
        period=td.period[rows].astype(int).tolist(),
        players_xy=_flat(td.xy[rows]),
        players_detected=None
        if td.detected is None
        else [bool(v) for v in td.detected[rows].ravel().tolist()],
        ball_xy=_flat(td.ball_xy[rows]),
        ball_detected=None
        if td.ball_detected is None
        else [
            bool(v)
            for v in (td.ball_detected[rows] & ~np.isnan(td.ball_xy[rows]).any(axis=1)).tolist()
        ],
        events=_events_out(ev),
        possession=poss,
        attack_sign=signs,
        source=_source(b),
    )


@app.get("/api/possessions/{possession_id}/replay", response_model=S.Replay)
def possession_replay(
    possession_id: str,
    pad_s: float = Query(1.0, ge=0, le=10, allow_inf_nan=False),
    step: int = Query(1, ge=1, le=25),
) -> S.Replay:
    match_id = possession_id.split(":", 1)[0]
    b = _bundle(match_id)
    hit = b.possessions.filter(pl.col("possession_id") == possession_id)
    if hit.height == 0:
        raise HTTPException(404, f"unknown possession {possession_id!r}")
    r = hit.row(0, named=True)
    pad = round(pad_s * b.tracking.fps)
    return replay(match_id, r["start_frame"] - pad, r["end_frame"] + pad, step, possession_id)


def _analysis_out(a: FrameAnalysis, horizon_s: float = 1.5) -> S.FrameAnalysisOut:
    st = a.state
    td = load_match(st.match_id).tracking
    row = td.index_of_frame(st.frame)
    k = round(horizon_s * td.fps)
    origin = td.xy[row - k] if row - k >= 0 and td.period[row - k] == st.period else None
    players = []
    for i, pid in enumerate(st.player_ids):
        ox = oy = None
        if origin is not None and not np.isnan(origin[i]).any():
            ox, oy = _f(origin[i, 0], 2), _f(origin[i, 1], 2)
        players.append(
            S.PlayerState(
                player_id=pid,
                team=st.teams[i],
                jersey=st.jerseys[i],
                x=_f(st.xy[i, 0], 2),
                y=_f(st.xy[i, 1], 2),
                vx=_f(st.vel[i, 0], 2),
                vy=_f(st.vel[i, 1], 2),
                is_gk=bool(st.is_gk[i]),
                position_provenance=st.position_provenance[i].value,
                velocity_provenance=st.velocity_provenance[i].value,
                reach_origin_x=ox,
                reach_origin_y=oy,
                reach_radius_m=REACH_SPEED_MPS * horizon_s if ox is not None else None,
            )
        )
    pc = None
    if a.control_grid is not None and a.grid is not None:
        g = a.grid
        pc = S.PitchControlOut(
            nx=g.nx,
            ny=g.ny,
            x0=-g.length / 2 + g.dx / 2,
            y0=-g.width / 2 + g.dy / 2,
            dx=g.dx,
            dy=g.dy,
            values=[round(float(v), 3) for v in np.nan_to_num(a.control_grid)],
        )
    opts = [
        S.PassOptionOut(
            receiver_id=o.receiver_id,
            target_x=round(o.target[0], 2),
            target_y=round(o.target[1], 2),
            distance_m=round(o.distance_m, 2),
            forward_progress_m=round(o.forward_progress_m, 2),
            ball_travel_time_s=round(o.ball_travel_time_s, 3),
            p_control_at_target=round(o.p_control_at_target, 4),
            p_lane_clear=round(o.p_lane_clear, 4),
            p_success_model=_f(o.p_success_model, 4),
            p_success=round(o.p_success, 4),
            lane_min_margin_s=_f(o.lane_min_margin_s),
            lane_closest_defender_id=o.lane_closest_defender_id,
            lane_closest_defender_perp_m=_f(o.lane_closest_defender_perp_m, 2),
            defenders_within_3m_of_lane=o.defenders_within_3m_of_lane,
        )
        for o in a.pass_options
    ]
    pr = None
    if a.pressure is not None:
        p = a.pressure
        pr = S.PressureOut(
            nearest_defender_id=p.nearest_defender_id,
            nearest_defender_distance_m=round(p.nearest_defender_distance_m, 2),
            closing_speed_mps=_f(p.closing_speed_mps, 2),
            defenders_within_5m=p.defenders_within_5m,
            defenders_within_10m=p.defenders_within_10m,
        )
    ball_nan = bool(np.isnan(st.ball).any())
    return S.FrameAnalysisOut(
        match_id=st.match_id,
        frame=st.frame,
        period=st.period,
        time_s=round(st.time_s, 2),
        period_time_s=round(float(_period_time(td, np.array([row]))[0]), 2),
        attacking_team=st.attacking_team,
        attack_sign=st.attack_sign,
        possession_id=st.possession_id,
        carrier_id=st.player_ids[st.carrier] if st.carrier is not None else None,
        ball_x=None if ball_nan else round(float(st.ball[0]), 2),
        ball_y=None if ball_nan else round(float(st.ball[1]), 2),
        ball_provenance=st.ball_provenance.value,
        players=players,
        pitch_control=pc,
        pass_options=opts,
        pressure=pr,
        control_area_m2=_f(a.control_area_m2, 1),
        control_area_final_third_m2=_f(a.control_area_final_third_m2, 1),
        model=S.ModelInfo(
            name="ppcf-time-to-intercept",
            reference="Spearman (2018) Beyond Expected Goals; Spearman et al. (2017)",
            status="literature parameters; not calibrated on this dataset",
            params=a.params.as_dict(),
        ),
        option_model=_option_model_info(st.match_id),
        warnings=a.warnings,
    )


def _provider_model(match_id: str) -> OptionModel | None:
    """The option model served for this match's provider, refusing another provider's model.

    Providers are never pooled (CLAUDE.md): a model whose metadata names a different provider,
    or no provider at all, is a packaging error, so the request fails instead of serving its
    probabilities. Every packaged model names its provider in ``meta.provider``.
    """
    m = load_default(match_id)
    provider = provider_of(match_id)
    if m is not None and m.meta.get("provider") != provider:
        raise HTTPException(
            503, f"option model for {provider!r} is labelled {m.meta.get('provider')!r}; refused"
        )
    return m


def _option_model_info(match_id: str) -> S.ModelInfo | None:
    m = _provider_model(match_id)
    if m is None:
        return None
    lomo = m.meta.get("lomo_pooled", {})
    provider = provider_of(match_id)
    return S.ModelInfo(
        name=str(m.meta.get("name", "option_model")),
        provider=provider,
        resource=PROVIDER_RESOURCES.get(provider),
        reference="FootyMind RQ1 option model (docs/RESEARCH.md)",
        status=(
            f"{m.meta.get('status', '')}; trained on {m.meta.get('n_train')} attempted passes; "
            f"held-out log loss {lomo.get('log_loss', float('nan')):.3f}, "
            f"AUC {lomo.get('auc', float('nan')):.3f}, ECE {lomo.get('ece', float('nan')):.3f}"
        ),
        params={
            "n_features": float(len(m.features)),
            "n_train": float(m.meta.get("n_train", 0)),
            "n_bootstrap": float(m.n_members),
            "lomo_log_loss": float(lomo.get("log_loss", float("nan"))),
            "lomo_auc": float(lomo.get("auc", float("nan"))),
            "lomo_ece": float(lomo.get("ece", float("nan"))),
        },
    )


def _frame_or_404(match_id: str, frame: int) -> None:
    b = _bundle(match_id)
    try:
        b.tracking.index_of_frame(frame)
    except KeyError as e:
        raise HTTPException(404, f"frame {frame} not in {match_id}") from e


@app.get("/api/matches/{match_id}/frames/{frame}/possession", response_model=S.PossessionSummary)
def frame_possession(match_id: str, frame: int) -> S.PossessionSummary:
    """The possession containing ``frame`` (else the latest one before it).

    Deep links keyed by (match, frame) stay valid when possession segmentation changes
    between pipeline versions, unlike links keyed by possession id.
    """
    _frame_or_404(match_id, frame)
    b = _bundle(match_id)
    _, pid = attacking_team_at(b, frame)
    if pid is None:
        raise HTTPException(404, f"no possession at or before frame {frame}")
    row = b.possessions.filter(pl.col("possession_id") == pid).row(0, named=True)
    return _possession_out(row, b.tracking.fps)


@app.get("/api/matches/{match_id}/frames/{frame}/analysis", response_model=S.FrameAnalysisOut)
def frame_analysis(match_id: str, frame: int, grid: bool = True) -> S.FrameAnalysisOut:
    _frame_or_404(match_id, frame)
    st = frame_state(match_id, frame)
    a = analyze(st, DEFAULT_PARAMS, DEFAULT_GRID if grid else None)
    out = _analysis_out(a)
    temporal = None
    if st.attacking_team is not None and st.carrier is not None:
        td = _bundle(match_id).tracking
        temporal = temporal_context(
            td, td.index_of_frame(frame), st.attacking_team, st.carrier, st.is_gk
        )
        out.temporal = {k: _f(v, 3) for k, v in temporal.items()}
    out.explanations = [S.StatementOut(**s.to_dict()) for s in explain(a, temporal)]
    return out


SIMILAR_DISCLAIMER = (
    "Neighbours are past passes from other matches of the same provider whose decision-time "
    "situation (positions, pressure, recent movement) was most similar. Their outcomes are "
    "observed. The share completed is a comparison, not a prediction for this pass."
)


def _similarity_validation(provider: str) -> dict[str, Any] | None:
    name = "similarity_validation_v1" + ("" if provider == "metrica" else f"_{provider}")
    p = EXPERIMENTS_DIR / "results" / name / "results.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or not isinstance(r.get("metrics", {}), dict):
            raise ValueError("not a results object")
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        # The served estimate is justified by this file: fail clearly instead of dropping it.
        raise HTTPException(
            503, f"validation results experiments/results/{name}/results.json unreadable: {e}"
        ) from e
    return {
        "ref": f"experiments/results/{name}/results.json",
        "knn_beats_base_rate": r.get("criterion_knn_beats_base_rate"),
        "log_loss": {k: v.get("log_loss") for k, v in r.get("metrics", {}).items()},
    }


@app.get("/api/matches/{match_id}/frames/{frame}/similar", response_model=S.SimilarMomentsOut)
def similar_moments(
    match_id: str, frame: int, k: int = Query(8, ge=1, le=50)
) -> S.SimilarMomentsOut:
    """The most similar past pass decisions (other matches, same provider) and their outcomes."""
    from footymind.research.similarity import (
        SIMILARITY_FEATURES,
        StaleIndexError,
        decision_row,
        index_path,
        load_index,
    )

    _frame_or_404(match_id, frame)
    provider = provider_of(match_id)
    try:
        idx = load_index(provider)
    except StaleIndexError as e:  # built from other bundles or another pipeline version
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # unreadable parquet or missing columns: polars raises many types
        raise HTTPException(
            503,
            f"similarity index {index_path(provider).name} is unreadable ({type(e).__name__}); "
            f"rebuild it with `footymind build-similarity-index --provider {provider}`",
        ) from e
    if idx is None:
        raise HTTPException(
            503, f"no similarity index for {provider}; run `footymind build-similarity-index`"
        )
    row = decision_row(match_id, frame)
    if row is None:
        raise HTTPException(404, "no pass decision starts within 2 s of this frame")
    x = np.array(
        [row[f] if row[f] is not None else np.nan for f in SIMILARITY_FEATURES], dtype=float
    )
    from footymind.research.similarity_validation import PRIOR_STRENGTH
    from footymind.research.similarity_validation import K as K_VALID

    validation = _similarity_validation(provider)
    got, dist = idx.query(x, exclude_match=match_id, k=k)
    # The validated estimate (C36) uses exactly the validated configuration: k = 50, shrunk.
    estimate = None
    wide, _ = idx.query(x, exclude_match=match_id, k=K_VALID)
    if wide.height == K_VALID:
        base = float(idx.rows["success"].mean())  # type: ignore[arg-type]
        succ = float(wide["success"].sum())
        estimate = {
            "k": K_VALID,
            "completion": round((succ + PRIOR_STRENGTH * base) / (K_VALID + PRIOR_STRENGTH), 3),
            "base_rate": round(base, 3),
            "method": f"share completed among the {K_VALID} most similar decisions from other "
            f"matches, shrunk toward the overall rate with {PRIOR_STRENGTH:g} pseudo-counts",
        }
    neighbours = [
        S.SimilarMoment(
            match_id=r["match_id"],
            frame=int(r["frame"]),
            possession_id=r["possession_id"],
            team=r["team"],
            passer_id=r["passer_id"],
            completed=bool(r["success"]),
            distance=round(float(d), 3),
            ball_x_att=_f(r["ball_x_att"], 1),
            press_nearest_def_m=_f(r["press_nearest_def_m"], 1),
            passer_control_s=_f(r["passer_control_s"], 1),
            period=int(r["period"]) if r.get("period") is not None else None,
            match_minute=_f(r.get("match_minute"), 1),
        )
        for r, d in zip(got.iter_rows(named=True), dist, strict=True)
    ]
    return S.SimilarMomentsOut(
        match_id=match_id,
        frame=frame,
        decision_frame=int(str(row["frame"])),
        passer_id=str(row["passer_id"]),
        k=len(neighbours),
        neighbours=neighbours,
        # No neighbours (e.g. an index of this match only): the share is unknown, not 0.
        completed_share=round(sum(n.completed for n in neighbours) / len(neighbours), 3)
        if neighbours
        else None,
        estimate=estimate,
        provider=provider,
        features=SIMILARITY_FEATURES,
        validation=validation,
        disclaimer=SIMILAR_DISCLAIMER,
    )


def _summary(d: dict[str, object], a: FrameAnalysis) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
    options_known = a.state.attacking_team is not None and not bool(np.isnan(a.state.ball).any())
    for k, v in d.items():
        if k.startswith("n_options") and not options_known:
            # Without a ball (or a possession) there are no options to count: unknown, not 0.
            out[k] = None
            continue
        if isinstance(v, float):
            out[k] = _f(v)
        elif v is None or isinstance(v, int | str):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# Defensive Shadow is validated per provider (never pooled). Each status cites its ledger row.
SHADOW_STATUS: dict[str, tuple[str, str, str]] = {
    "metrica": (
        "experimental: partly validated on Metrica (C17): about 2x chance at ranking the "
        "eventual interceptor, but not better than nearest-defender. Not a defender rating.",
        "C17",
        "experiments/results/shadow_validation_v1/results.json",
    ),
    "skillcorner": (
        "experimental: not validated on SkillCorner (C29): no better than chance on broadcast "
        "data at ranking the eventual interceptor; worse than nearest-defender. "
        "Not a defender rating.",
        "C29",
        "experiments/results/shadow_validation_v1_skillcorner/results.json",
    ),
}
SHADOW_STATUS_UNVALIDATED = "experimental: not validated for this provider. Not a defender rating."


@app.get(
    "/api/matches/{match_id}/frames/{frame}/defensive-shadow", response_model=S.DefensiveShadowOut
)
def frame_defensive_shadow(
    match_id: str,
    frame: int,
    horizon_s: float = Query(1.0, gt=0, le=3, allow_inf_nan=False),
    value_model: Literal["option_model", "physics"] = "option_model",
) -> S.DefensiveShadowOut:
    _frame_or_404(match_id, frame)
    try:
        r = defensive_shadow(match_id, frame, horizon_s=horizon_s, value_model=value_model)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    provider = provider_of(match_id)
    status, claim, ref = SHADOW_STATUS.get(provider, (SHADOW_STATUS_UNVALIDATED, None, None))
    return S.DefensiveShadowOut(
        provider=provider,
        status=status,
        validation_claim=claim,
        validation_ref=ref,
        match_id=r.match_id,
        frame=r.frame,
        attacking_team=r.attacking_team,  # type: ignore[arg-type]
        carrier_id=r.carrier_id,
        value_model=r.value_model,
        v_best_observed=round(r.v_best_observed, 4),
        v_sum_observed=round(r.v_sum_observed, 4),
        defenders=[
            S.DefenderShadowOut(
                player_id=d.player_id,
                x=round(d.x, 2),
                y=round(d.y, 2),
                shadow_best=round(d.shadow_best, 4) + 0.0,
                shadow_sum=round(d.shadow_sum, 4) + 0.0,
                alt_radius_m=d.alt_radius_m,
                alternatives=d.alternatives,
                most_suppressed_receiver=d.most_suppressed_receiver,
                most_suppressed_delta=round(d.most_suppressed_delta, 4) + 0.0,
                distinguishable_from_noise=abs(d.shadow_sum) >= SHADOW_NOISE_FLOOR,
            )
            for d in r.defenders
        ],
        assumptions=r.assumptions,
        noise_floor=SHADOW_NOISE_FLOOR,
        horizon_s=horizon_s,
        alt_radius_m=r.defenders[0].alt_radius_m if r.defenders else 0.0,
    )


def _policy_out(p: PolicyInfo) -> S.ResponsePolicyOut:
    return S.ResponsePolicyOut(
        name=p.name,  # type: ignore[arg-type]
        description=p.description,
        assumptions=p.assumptions,
        validated=p.validated,
        validation_ref=p.validation_ref,
        status=p.status,
    )


@app.get("/api/response-policies", response_model=list[S.ResponsePolicyOut])
def response_policies(match_id: str | None = None) -> list[S.ResponsePolicyOut]:
    """Opponent response policies for scenarios.

    With ``match_id``, ``validated`` refers to that match's provider. Without it, a policy
    counts as validated only if it is validated for every provider.
    """
    provider = None
    if match_id is not None:
        if match_id not in available_matches():
            raise HTTPException(404, f"unknown match {match_id!r}")
        provider = provider_of(match_id)
    return [_policy_out(policy_info(n, provider)) for n in POLICY_NAMES]


DISCLAIMER = (
    "Model-based counterfactual under stated assumptions. It shows how the spatial model's "
    "estimates change, not what would have happened in the match."
)


@app.post("/api/scenarios", response_model=S.ScenarioResponse)
def scenario(req: S.ScenarioRequest) -> S.ScenarioResponse:
    _frame_or_404(req.match_id, req.frame)
    _provider_model(req.match_id)
    try:
        r = run_scenario(
            req.match_id,
            req.frame,
            [Intervention(i.player_id, i.x, i.y) for i in req.interventions],
            policy=req.response_policy,
            follow_fraction=req.follow_fraction,
            velocity_mode=req.velocity_mode,
            horizon_s=req.horizon_s,
            ensemble=req.ensemble,
            seed=req.seed,
            value_model=req.value_model,
        )
    except KeyError as e:
        raise HTTPException(404, f"unknown player {e}") from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    def interval(t: tuple[float, float] | None, k: int) -> float | None:
        return round(t[k], 4) if t else None

    return S.ScenarioResponse(
        observed=_analysis_out(r.observed, req.horizon_s),
        hypothetical=_analysis_out(r.hypothetical, req.horizon_s),
        feasibility=[
            S.FeasibilityOut(
                player_id=f.player_id,
                displacement_m=round(f.displacement_m, 2),
                reachable_radius_m=f.reachable_radius_m,
                reachable=f.reachable,
                origin_x=_f(f.origin_xy[0], 2) if f.origin_xy else None,
                origin_y=_f(f.origin_xy[1], 2) if f.origin_xy else None,
                inside_pitch=f.inside_pitch,
                min_separation_m=round(f.min_separation_m, 2),
                supported=f.supported,
                kind=f.kind,  # type: ignore[arg-type]
                kinematic_reachable=f.kinematic_reachable,
                notes=f.notes,
            )
            for f in r.feasibility
        ],
        movements=[
            S.MovementOut(
                player_id=m.player_id,
                from_x=round(m.from_xy[0], 2),
                from_y=round(m.from_xy[1], 2),
                to_x=round(m.to_xy[0], 2),
                to_y=round(m.to_xy[1], 2),
                provenance=m.provenance.value,
                reason=m.reason,
            )
            for m in r.movements
        ],
        option_deltas=[
            S.OptionDeltaOut(
                receiver_id=d.receiver_id,
                observed_p=_f(d.observed_p, 4),
                hypothetical_p=_f(d.hypothetical_p, 4),
                delta=_f(d.delta, 4),
                observed_p10=interval(d.observed_p_interval, 0),
                observed_p90=interval(d.observed_p_interval, 1),
                hypothetical_p10=interval(d.hypothetical_p_interval, 0),
                hypothetical_p90=interval(d.hypothetical_p_interval, 1),
                delta_sign_agreement=_f(d.delta_sign_agreement, 3),
            )
            for d in r.option_deltas
        ],
        summary_observed=_summary(r.summary_observed, r.observed),
        summary_hypothetical=_summary(r.summary_hypothetical, r.hypothetical),
        ensemble_size=r.ensemble_size,
        supported=r.supported,
        warnings=r.warnings,
        assumptions=r.assumptions,
        disclaimer=DISCLAIMER,
        value_model=r.value_model,
        policy=_policy_out(r.policy),
    )


_web_dist = Path(os.environ.get("FOOTYMIND_WEB_DIST", REPO_ROOT / "web" / "dist"))
if _web_dist.exists():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")
