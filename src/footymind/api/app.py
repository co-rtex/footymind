"""FootyMind HTTP API (FastAPI). Run with ``footymind serve``."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from footymind import __version__
from footymind.api import schemas as S
from footymind.counterfactual.scenario import REACH_SPEED_MPS, Intervention, run_scenario
from footymind.data.store import MatchBundle, StaleBundleError, available_matches, load_match
from footymind.ghost.defensive_shadow import defensive_shadow
from footymind.paths import REPO_ROOT
from footymind.spatial.analysis import FrameAnalysis, analyze
from footymind.spatial.option_model import load_default
from footymind.spatial.pitch_control import DEFAULT_GRID, DEFAULT_PARAMS
from footymind.spatial.state import frame_state

MAX_REPLAY_FRAMES = 4000

app = FastAPI(
    title="FootyMind API",
    version=__version__,
    description=(
        "Tracking-based football analysis: observed replay, spatial analytics and "
        "constrained counterfactual scenarios. All analytical outputs carry provenance labels."
    ),
)
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
    b = _bundle(match_id)
    q = b.manifest["quality"]["tracking"]
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
    return S.MatchSummary(
        match_id=match_id,
        source=_source(b),
        fps=b.tracking.fps,
        n_frames=b.tracking.n_frames,
        periods=periods,
        n_possessions=b.possessions.height,
        n_events=b.events.height,
        pitch=S.Pitch(length=b.tracking.pitch.length, width=b.tracking.pitch.width),
    )


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


def _flat(a: np.ndarray, nd: int = 2) -> list[float | None]:
    r = np.round(a.astype(np.float64).ravel(), nd)
    return [None if math.isnan(v) else float(v) for v in r.tolist()]


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
        period=td.period[rows].astype(int).tolist(),
        players_xy=_flat(td.xy[rows]),
        ball_xy=_flat(td.ball_xy[rows]),
        events=_events_out(ev),
        possession=poss,
        attack_sign=signs,
        source=_source(b),
    )


@app.get("/api/possessions/{possession_id}/replay", response_model=S.Replay)
def possession_replay(
    possession_id: str, pad_s: float = Query(1.0, ge=0, le=10), step: int = Query(1, ge=1, le=25)
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
        attacking_team=st.attacking_team,
        attack_sign=st.attack_sign,
        possession_id=st.possession_id,
        carrier_id=st.player_ids[st.carrier] if st.carrier is not None else None,
        ball_x=None if ball_nan else round(float(st.ball[0]), 2),
        ball_y=None if ball_nan else round(float(st.ball[1]), 2),
        ball_provenance="observed",
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
        option_model=_option_model_info(),
        warnings=a.warnings,
    )


def _option_model_info() -> S.ModelInfo | None:
    m = load_default()
    if m is None:
        return None
    lomo = m.meta.get("lomo_pooled", {})
    return S.ModelInfo(
        name=str(m.meta.get("name", "option_model")),
        reference="FootyMind RQ1 option model (docs/RESEARCH.md)",
        status=(
            f"{m.meta.get('status', '')}; trained on {m.meta.get('n_train')} attempted passes; "
            f"held-out log loss {lomo.get('log_loss', float('nan')):.3f}, "
            f"AUC {lomo.get('auc', float('nan')):.3f}, ECE {lomo.get('ece', float('nan')):.3f}"
        ),
        params={"n_features": float(len(m.features))},
    )


def _frame_or_404(match_id: str, frame: int) -> None:
    b = _bundle(match_id)
    try:
        b.tracking.index_of_frame(frame)
    except KeyError as e:
        raise HTTPException(404, f"frame {frame} not in {match_id}") from e


@app.get("/api/matches/{match_id}/frames/{frame}/analysis", response_model=S.FrameAnalysisOut)
def frame_analysis(match_id: str, frame: int, grid: bool = True) -> S.FrameAnalysisOut:
    _frame_or_404(match_id, frame)
    a = analyze(frame_state(match_id, frame), DEFAULT_PARAMS, DEFAULT_GRID if grid else None)
    return _analysis_out(a)


def _summary(d: dict[str, object]) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = _f(v)
        elif v is None or isinstance(v, int | str):
            out[k] = v
        else:
            out[k] = str(v)
    return out


@app.get(
    "/api/matches/{match_id}/frames/{frame}/defensive-shadow", response_model=S.DefensiveShadowOut
)
def frame_defensive_shadow(
    match_id: str,
    frame: int,
    horizon_s: float = Query(1.0, gt=0, le=3),
    value_model: Literal["option_model", "physics"] = "option_model",
) -> S.DefensiveShadowOut:
    _frame_or_404(match_id, frame)
    try:
        r = defensive_shadow(match_id, frame, horizon_s=horizon_s, value_model=value_model)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return S.DefensiveShadowOut(
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
            )
            for d in r.defenders
        ],
        assumptions=r.assumptions,
    )


DISCLAIMER = (
    "Model-based counterfactual under stated assumptions. It shows how the spatial model's "
    "estimates change, not what would have happened in the match."
)


@app.post("/api/scenarios", response_model=S.ScenarioResponse)
def scenario(req: S.ScenarioRequest) -> S.ScenarioResponse:
    _frame_or_404(req.match_id, req.frame)
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
                origin_x=f.origin_xy[0] if f.origin_xy else None,
                origin_y=f.origin_xy[1] if f.origin_xy else None,
                inside_pitch=f.inside_pitch,
                min_separation_m=round(f.min_separation_m, 2),
                supported=f.supported,
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
        summary_observed=_summary(r.summary_observed),
        summary_hypothetical=_summary(r.summary_hypothetical),
        ensemble_size=r.ensemble_size,
        supported=r.supported,
        warnings=r.warnings,
        assumptions=r.assumptions,
        disclaimer=DISCLAIMER,
        value_model=r.value_model,
    )


_web_dist = Path(os.environ.get("FOOTYMIND_WEB_DIST", REPO_ROOT / "web" / "dist"))
if _web_dist.exists():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")
