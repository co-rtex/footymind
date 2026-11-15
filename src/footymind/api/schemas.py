"""Typed API contract. The frontend mirrors these in ``web/src/api/types.ts``.

Conventions:
* Coordinates are canonical metres: origin at the centre spot, x along the
  length, y up (see docs/ARCHITECTURE.md).
* Flat arrays are row-major. ``null`` encodes an unobserved value. Values are
  never fabricated.
* Every analytical quantity has a provenance label: observed, derived,
  estimated, hypothetical or simulated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ProvenanceLabel = Literal["observed", "derived", "estimated", "hypothetical", "simulated"]
ResponsePolicyName = Literal[
    "none",
    "marker_follows",
    "press_carrier",
    "zonal_shift",
    "lane_cover",
    "learned",
    "learned_innovation",
]
Team = Literal["home", "away"]


class Pitch(BaseModel):
    length: float
    width: float


class SourceInfo(BaseModel):
    provider: str
    name: str
    url: str
    license_note: str


class PeriodInfo(BaseModel):
    period: int
    first_frame: int
    last_frame: int
    duration_s: float
    home_attacks_positive_x: bool


class MatchSummary(BaseModel):
    match_id: str
    source: SourceInfo
    fps: float
    n_frames: int
    periods: list[PeriodInfo]
    n_possessions: int
    n_events: int
    pitch: Pitch
    home_label: str = "Home"
    away_label: str = "Away"
    competition: str | None = Field(
        default=None, description="provider metadata (SkillCorner only; Metrica is anonymised)"
    )
    date: str | None = Field(default=None, description="kick-off date (UTC), provider metadata")


class PossessionSummary(BaseModel):
    possession_id: str
    team: Team
    period: int
    start_frame: int
    end_frame: int
    start_time_s: float
    duration_s: float
    n_events: int
    n_passes: int
    first_type: str | None
    first_subtype: str | None
    last_type: str | None
    last_subtype: str | None
    has_shot: bool
    has_goal: bool


class EventOut(BaseModel):
    event_id: int
    team: Team
    type: str
    subtype: str | None
    period: int
    start_frame: int
    end_frame: int | None
    start_time_s: float
    from_player_id: str | None
    to_player_id: str | None
    start_x: float | None
    start_y: float | None
    end_x: float | None
    end_y: float | None
    possession_id: str | None


class PlayerOut(BaseModel):
    player_id: str
    team: Team
    jersey: int


class Replay(BaseModel):
    """Observed tracking for a frame window.

    ``players_xy`` has length ``n_frames * n_players * 2``, row-major in
    ``[frame][player][x|y]``, with ``null`` when not observed. ``ball_xy`` has
    length ``n_frames * 2``.
    """

    match_id: str
    fps: float = Field(description="provider frame rate")
    step: int = Field(description="frame subsampling step (1 = every frame)")
    pitch: Pitch
    players: list[PlayerOut]
    frame: list[int]
    time_s: list[float]
    period_time_s: list[float] = Field(
        default_factory=list,
        description="seconds since the first tracked frame of the period (match clock basis)",
    )
    period: list[int]
    players_xy: list[float | None]
    players_detected: list[bool] | None = Field(
        None,
        description=(
            "length n_frames * n_players, row-major [frame][player]; false where the provider "
            "extrapolated a position outside the camera view (provenance 'estimated') and also "
            "where the position is null (player absent); null when every non-null position is "
            "observed (optical tracking). time_s is provider time: build a match clock from "
            "MatchSummary.periods[].first_frame (SkillCorner periods do not start at 0)"
        ),
    )
    ball_xy: list[float | None]
    ball_detected: list[bool] | None = Field(
        None,
        description=(
            "length n_frames, aligned with ball_xy; false where the provider extrapolated the "
            "ball (provenance 'estimated') and where the ball position is null; null when the "
            "provider has no ball detection mask (optical tracking, or bundles ingested before "
            "the mask existed)"
        ),
    )
    events: list[EventOut]
    possession: PossessionSummary | None
    attack_sign: dict[str, dict[str, int]] = Field(
        description="period -> team -> +1 if the team attacks towards +x"
    )
    provenance: ProvenanceLabel = "observed"
    source: SourceInfo


class PlayerState(BaseModel):
    player_id: str
    team: Team
    jersey: int
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    is_gk: bool
    position_provenance: ProvenanceLabel
    velocity_provenance: ProvenanceLabel
    reach_origin_x: float | None = Field(
        None, description="observed position horizon_s earlier, or null"
    )
    reach_origin_y: float | None = None
    reach_radius_m: float | None = None


class PitchControlOut(BaseModel):
    nx: int
    ny: int
    x0: float = Field(description="x of the first cell centre")
    y0: float = Field(description="y of the first cell centre")
    dx: float
    dy: float
    values: list[float] = Field(
        description="attacking-team control probability, row-major (ny rows from y=-W/2 upwards)"
    )
    provenance: ProvenanceLabel = "estimated"


class PassOptionOut(BaseModel):
    receiver_id: str
    target_x: float
    target_y: float
    distance_m: float
    forward_progress_m: float
    ball_travel_time_s: float
    p_control_at_target: float
    p_lane_clear: float
    p_success: float = Field(description="physics heuristic: p_control_at_target * p_lane_clear")
    p_success_model: float | None = Field(
        None, description="data-calibrated option model (see model.option_model)"
    )
    lane_min_margin_s: float | None
    lane_closest_defender_id: str | None
    lane_closest_defender_perp_m: float | None
    defenders_within_3m_of_lane: int
    provenance: ProvenanceLabel = "estimated"


class PressureOut(BaseModel):
    nearest_defender_id: str | None
    nearest_defender_distance_m: float
    closing_speed_mps: float | None
    defenders_within_5m: int
    defenders_within_10m: int
    provenance: ProvenanceLabel = "derived"


class ModelInfo(BaseModel):
    name: str
    reference: str
    status: str
    params: dict[str, float]
    provider: str | None = Field(
        default=None,
        description="tracking provider the model was trained on (option model only; models are "
        "per provider and never pooled)",
    )
    resource: str | None = Field(default=None, description="packaged model file")


class StatementOut(BaseModel):
    """One evidence-grounded explanation statement (deterministic; see spatial/explain.py)."""

    kind: Literal["pressure", "trend", "option", "caveat"]
    text: str
    metric: str = Field(description="the computed metric this statement is generated from")
    value: float | int | str | None
    provenance: ProvenanceLabel
    players: list[str]


class SimilarMoment(BaseModel):
    match_id: str
    frame: int
    possession_id: str | None
    team: Team
    passer_id: str
    completed: bool = Field(description="OBSERVED outcome of that pass")
    distance: float = Field(description="standardised feature distance (smaller = more similar)")
    ball_x_att: float | None
    press_nearest_def_m: float | None
    passer_control_s: float | None
    period: int | None = None
    match_minute: float | None = Field(
        default=None, description="provider match clock in minutes at the pass start"
    )


class SimilarMomentsOut(BaseModel):
    match_id: str
    frame: int = Field(description="the requested frame")
    decision_frame: int = Field(description="start frame of the pass decision that was matched")
    passer_id: str
    k: int
    neighbours: list[SimilarMoment]
    completed_share: float | None = Field(
        description="share of the listed neighbours completed; null when none are listed"
    )
    estimate: dict[str, Any] | None = Field(
        default=None,
        description="the validated configuration: completion rate among the 50 nearest "
        "decisions, shrunk toward the index base rate (C36); null if the index is too small",
    )
    provider: str
    features: list[str]
    validation: dict[str, Any] | None = Field(
        default=None, description="held-out validation summary for this provider, if available"
    )
    disclaimer: str


class FrameAnalysisOut(BaseModel):
    match_id: str
    frame: int
    period: int
    time_s: float
    period_time_s: float = Field(0.0, description="seconds since the period's first tracked frame")
    reach_horizon_s: float = Field(
        1.5, description="horizon of each player's reach_origin and reach_radius_m, in seconds"
    )
    reach_speed_mps: float = Field(
        7.0, description="speed bound behind reach_radius_m (= reach_speed_mps * reach_horizon_s)"
    )
    attacking_team: Team | None
    attack_sign: int
    possession_id: str | None
    carrier_id: str | None
    ball_x: float | None
    ball_y: float | None
    ball_provenance: ProvenanceLabel
    players: list[PlayerState]
    pitch_control: PitchControlOut | None
    pass_options: list[PassOptionOut]
    pressure: PressureOut | None
    control_area_m2: float | None
    control_area_final_third_m2: float | None
    model: ModelInfo
    option_model: ModelInfo | None = None
    warnings: list[str]
    temporal: dict[str, float | None] | None = Field(
        default=None,
        description="causal temporal context for the carrier (DERIVED; frames at or before this "
        "frame only). Observed frames only; null for hypothetical states.",
    )
    explanations: list[StatementOut] = Field(
        default_factory=list,
        description="evidence-grounded statements, each tied to one computed metric",
    )


class InterventionIn(BaseModel):
    player_id: str
    x: float = Field(ge=-60, le=60, allow_inf_nan=False)
    y: float = Field(ge=-40, le=40, allow_inf_nan=False)


class ScenarioRequest(BaseModel):
    match_id: str
    frame: int
    interventions: list[InterventionIn] = Field(min_length=1, max_length=4)
    response_policy: ResponsePolicyName = "none"
    follow_fraction: float = Field(0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    velocity_mode: Literal["keep", "zero"] = "keep"
    horizon_s: float = Field(1.5, gt=0.0, le=5.0, allow_inf_nan=False)
    ensemble: int = Field(24, ge=0, le=64)
    seed: int = 0
    value_model: Literal["option_model", "physics"] = "option_model"

    @model_validator(mode="after")
    def _one_edit_per_player(self) -> ScenarioRequest:
        ids = [i.player_id for i in self.interventions]
        dup = sorted({p for p in ids if ids.count(p) > 1})
        if dup:
            raise ValueError(f"more than one intervention for {', '.join(dup)}")
        return self


class FeasibilityOut(BaseModel):
    player_id: str
    displacement_m: float
    reachable_radius_m: float
    reachable: bool
    origin_x: float | None
    origin_y: float | None
    inside_pitch: bool
    min_separation_m: float
    supported: bool
    notes: list[str]
    kind: Literal["user", "simulated"] = "user"
    kinematic_reachable: bool | None = Field(
        None,
        description=(
            "inside the kinematic envelope (reaction delay, acceleration and speed limits) "
            "from the observed position and velocity horizon_s earlier; null if that origin "
            "is unobserved. Diagnostic only: the a priori envelope is tight and excludes a "
            "sizeable share of real defender positions (docs/RESEARCH.md, response policies), "
            "so it does not change 'supported'"
        ),
    )


class FittedRangeOut(BaseModel):
    share: float = Field(description="share of real rows below limit_m (e.g. 0.99)")
    limit_m: float = Field(description="relocations larger than this are extrapolations")
    horizon_s: float
    quantity: str
    ref: str = Field(description="committed results.json the limit is read from")


class ResponsePolicyOut(BaseModel):
    name: ResponsePolicyName
    description: str
    assumptions: list[str]
    validated: bool = Field(
        description=(
            "true only if the policy's short-horizon forecasts of real defender movement beat "
            "constant velocity on held-out matches of this provider (learned_innovation: also "
            "a permuted placebo, given the attackers' real movement). Never a validation of "
            "responses to hypothetical interventions (no ground truth exists)"
        )
    )
    validation_ref: str | None = Field(
        description="committed results.json of the validation run, or null if never validated"
    )
    status: str = Field(description="'validated: ...', 'experimental: ...' or 'no response'")
    verdict: Literal[
        "no_response",
        "forecast_validated",
        "observational_validated",
        "observational_supported",
        "not_supported",
        "unchecked",
    ] = Field(
        description=(
            "machine-readable form of status: forecast_validated (beats constant velocity on "
            "held-out real play), observational_validated (also beats a placebo given the "
            "attackers' real movement), observational_supported (a rule consistent with real "
            "reactions in the paired observational check, C70), not_supported, unchecked, or "
            "no_response (the static baseline). Never an intervention validation"
        )
    )
    fitted_range: FittedRangeOut | None = Field(
        None,
        description=(
            "for a linear, data-fitted response: the range of real movement it was fitted on "
            "(for this match's provider; without a match, the tightest over providers)"
        ),
    )


class MovementOut(BaseModel):
    player_id: str
    from_x: float
    from_y: float
    to_x: float
    to_y: float
    provenance: ProvenanceLabel
    reason: str


class OptionDeltaOut(BaseModel):
    receiver_id: str
    observed_p: float | None
    hypothetical_p: float | None
    delta: float | None
    observed_p10: float | None
    observed_p90: float | None
    hypothetical_p10: float | None
    hypothetical_p90: float | None
    delta_sign_agreement: float | None = Field(
        description="share of the parameter ensemble whose delta has the same sign"
    )


class ScenarioResponse(BaseModel):
    observed: FrameAnalysisOut
    hypothetical: FrameAnalysisOut
    feasibility: list[FeasibilityOut]
    movements: list[MovementOut]
    option_deltas: list[OptionDeltaOut]
    summary_observed: dict[str, float | int | str | None]
    summary_hypothetical: dict[str, float | int | str | None]
    ensemble_size: int
    supported: bool
    warnings: list[str]
    assumptions: list[str]
    disclaimer: str
    value_model: Literal["option_model", "physics"] = Field(
        description="which per-option probability the deltas use"
    )
    policy: ResponsePolicyOut = Field(description="the opponent response policy that was applied")


class DefenderShadowOut(BaseModel):
    player_id: str
    x: float
    y: float
    shadow_best: float = Field(description="mean alt V_best - observed V_best (+ = suppressive)")
    shadow_sum: float = Field(description="mean alt V_sum - observed V_sum (+ = suppressive)")
    alt_radius_m: float
    alternatives: list[tuple[float, float]]
    most_suppressed_receiver: str | None
    most_suppressed_delta: float
    distinguishable_from_noise: bool = Field(
        description="|shadow_sum| above the tracking-noise floor (shadow_stability_v1)"
    )
    provenance: ProvenanceLabel = "estimated"


class DefensiveShadowOut(BaseModel):
    match_id: str
    frame: int
    attacking_team: Team
    carrier_id: str | None
    value_model: Literal["option_model", "physics"]
    v_best_observed: float
    v_sum_observed: float
    defenders: list[DefenderShadowOut]
    assumptions: list[str]
    noise_floor: float = Field(description="|shadow_sum| below this is within tracking noise")
    horizon_s: float
    alt_radius_m: float
    provider: str
    status: str = Field(description="validation status for this match's provider")
    validation_claim: str | None = Field(
        description="claims-ledger id behind the status (docs/CLAIMS_LEDGER.md); null if none"
    )
    validation_ref: str | None = Field(description="committed results file behind the claim")
