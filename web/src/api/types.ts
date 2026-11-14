/**
 * Typed mirror of the FastAPI contract in `src/footymind/api/schemas.py`.
 *
 * Keep this file in lock-step with the Python models: same field names, same
 * nullability. Pydantic `X | None` maps to `X | null` here. Fields with a
 * Python default are still always serialised by FastAPI, so they are required.
 *
 * Conventions (from the backend):
 * - coordinates are canonical metres, origin at the centre spot, x along the
 *   105 m length, y UP across the 68 m width;
 * - flat arrays are row-major and `null` encodes "not observed";
 * - every analytical quantity carries a provenance label.
 */

export type ProvenanceLabel = "observed" | "derived" | "estimated" | "hypothetical" | "simulated";
export type Team = "home" | "away";

export interface Pitch {
  length: number;
  width: number;
}

export interface SourceInfo {
  provider: string;
  name: string;
  url: string;
  license_note: string;
}

export interface PeriodInfo {
  period: number;
  first_frame: number;
  last_frame: number;
  duration_s: number;
  home_attacks_positive_x: boolean;
}

export interface MatchSummary {
  match_id: string;
  source: SourceInfo;
  fps: number;
  n_frames: number;
  periods: PeriodInfo[];
  n_possessions: number;
  n_events: number;
  pitch: Pitch;
  /** team short name ("Brisbane FC"); "Home" for anonymised providers (Metrica) */
  home_label: string;
  away_label: string;
  /** "A-League, Round 5" (provider metadata); null for anonymised providers, absent on older backends */
  competition?: string | null;
  /** kick-off date "2024-11-23"; null for anonymised providers, absent on older backends */
  date?: string | null;
}

export interface PossessionSummary {
  possession_id: string;
  team: Team;
  period: number;
  start_frame: number;
  end_frame: number;
  start_time_s: number;
  duration_s: number;
  n_events: number;
  n_passes: number;
  first_type: string | null;
  first_subtype: string | null;
  last_type: string | null;
  last_subtype: string | null;
  has_shot: boolean;
  has_goal: boolean;
}

export interface EventOut {
  event_id: number;
  team: Team;
  type: string;
  subtype: string | null;
  period: number;
  start_frame: number;
  end_frame: number | null;
  start_time_s: number;
  from_player_id: string | null;
  to_player_id: string | null;
  start_x: number | null;
  start_y: number | null;
  end_x: number | null;
  end_y: number | null;
  possession_id: string | null;
}

export interface PlayerOut {
  player_id: string;
  team: Team;
  jersey: number;
}

/** Observed tracking for a frame window (wire format). */
export interface Replay {
  match_id: string;
  /** provider frame rate */
  fps: number;
  /** frame subsampling step (1 = every frame) */
  step: number;
  pitch: Pitch;
  players: PlayerOut[];
  frame: number[];
  time_s: number[];
  /** seconds since the first tracked frame of each frame's period (match clock basis); absent on older backends */
  period_time_s?: number[];
  period: number[];
  /** length n_frames * n_players * 2, row-major [frame][player][x|y], null = not observed */
  players_xy: (number | null)[];
  /**
   * length n_frames * n_players, row-major [frame][player]; false where the provider
   * extrapolated a position outside the camera view (provenance "estimated");
   * null when every non-null position is observed (optical tracking)
   */
  players_detected?: boolean[] | null;
  /** length n_frames * 2 */
  ball_xy: (number | null)[];
  /**
   * length n_frames; false where the provider extrapolated the ball (provenance "estimated");
   * null or absent when every non-null ball position is observed (optical tracking, older backends)
   */
  ball_detected?: boolean[] | null;
  events: EventOut[];
  possession: PossessionSummary | null;
  /** period -> team -> +1 if the team attacks towards +x */
  attack_sign: Record<string, Record<string, number>>;
  provenance: ProvenanceLabel;
  source: SourceInfo;
}

export interface PlayerState {
  player_id: string;
  team: Team;
  jersey: number;
  x: number | null;
  y: number | null;
  vx: number | null;
  vy: number | null;
  is_gk: boolean;
  position_provenance: ProvenanceLabel;
  velocity_provenance: ProvenanceLabel;
  /** observed position horizon_s earlier, or null */
  reach_origin_x: number | null;
  reach_origin_y: number | null;
  reach_radius_m: number | null;
}

export interface PitchControlOut {
  nx: number;
  ny: number;
  /** x of the first cell centre */
  x0: number;
  /** y of the first cell centre */
  y0: number;
  dx: number;
  dy: number;
  /** attacking-team control probability, row-major (ny rows from y=-W/2 upwards) */
  values: number[];
  provenance: ProvenanceLabel;
}

export interface PassOptionOut {
  receiver_id: string;
  target_x: number;
  target_y: number;
  distance_m: number;
  forward_progress_m: number;
  ball_travel_time_s: number;
  p_control_at_target: number;
  p_lane_clear: number;
  /** physics heuristic: p_control_at_target * p_lane_clear (uncalibrated) */
  p_success: number;
  /** data-calibrated option model (see FrameAnalysisOut.option_model); null if unavailable */
  p_success_model: number | null;
  lane_min_margin_s: number | null;
  lane_closest_defender_id: string | null;
  lane_closest_defender_perp_m: number | null;
  defenders_within_3m_of_lane: number;
  provenance: ProvenanceLabel;
}

export interface PressureOut {
  nearest_defender_id: string | null;
  nearest_defender_distance_m: number;
  closing_speed_mps: number | null;
  defenders_within_5m: number;
  defenders_within_10m: number;
  provenance: ProvenanceLabel;
}

export interface ModelInfo {
  name: string;
  reference: string;
  status: string;
  params: Record<string, number>;
  /** provider the served model belongs to (never another provider's model); null for the physics model */
  provider?: string | null;
  /** packaged model file; null for the physics model */
  resource?: string | null;
}

export interface FrameAnalysisOut {
  match_id: string;
  frame: number;
  period: number;
  time_s: number;
  /** seconds since the period's first tracked frame (absent on older backends) */
  period_time_s?: number;
  /** horizon (s) and speed bound (m/s) behind each player's reach_origin and reach_radius_m (newer backends) */
  reach_horizon_s?: number;
  reach_speed_mps?: number;
  attacking_team: Team | null;
  attack_sign: number;
  possession_id: string | null;
  carrier_id: string | null;
  ball_x: number | null;
  ball_y: number | null;
  ball_provenance: ProvenanceLabel;
  players: PlayerState[];
  pitch_control: PitchControlOut | null;
  pass_options: PassOptionOut[];
  pressure: PressureOut | null;
  control_area_m2: number | null;
  control_area_final_third_m2: number | null;
  model: ModelInfo;
  /** calibrated option model behind PassOptionOut.p_success_model; status holds held-out metrics and caveats */
  option_model: ModelInfo | null;
  warnings: string[];
  /**
   * Causal last-second features for the carrier (derived from frames at or before this one).
   * Null for hypothetical states; absent on older backends.
   */
  temporal?: Record<string, number | null> | null;
  /** Evidence-grounded statements, each tied to one computed metric (absent on older backends). */
  explanations?: StatementOut[];
}

export type StatementKind = "pressure" | "trend" | "option" | "caveat";

/** One "why this moment" statement (mirrors StatementOut). The text is shown as given. */
export interface StatementOut {
  kind: StatementKind | (string & {});
  text: string;
  /** the computed metric the statement is generated from */
  metric: string;
  value: number | string | null;
  provenance: ProvenanceLabel;
  /** players the statement is about, highlighted on the pitch when the statement is focused */
  players: string[];
}

/** One past pass decision from another match (mirrors SimilarMoment). */
export interface SimilarMoment {
  match_id: string;
  frame: number;
  possession_id: string | null;
  team: Team;
  passer_id: string;
  /** OBSERVED outcome of that pass */
  completed: boolean;
  /** standardised feature distance (smaller = more similar) */
  distance: number;
  /** ball x in the attacking direction (m from halfway, negative = own half) */
  ball_x_att: number | null;
  press_nearest_def_m: number | null;
  passer_control_s: number | null;
  /** period of that pass (absent on older backends) */
  period?: number | null;
  /** provider match clock in minutes at the pass start, 0.1 min resolution (absent on older backends) */
  match_minute?: number | null;
}

export interface SimilarValidation {
  ref: string | null;
  /** pre-registered criterion: does the neighbour outcome beat the base rate on held-out matches */
  knn_beats_base_rate: boolean | null;
  log_loss: unknown;
}

/** GET /api/matches/{mid}/frames/{frame}/similar (mirrors SimilarMomentsOut). 404: no pass decision near the frame; 503: no index. */
export interface SimilarMomentsOut {
  match_id: string;
  frame: number;
  decision_frame: number;
  passer_id: string;
  k: number;
  neighbours: SimilarMoment[];
  /** null when there are no neighbours */
  completed_share: number | null;
  /** The validated configuration (C36): k = 50, shrunk toward the base rate. Optional. */
  estimate?: { k: number; completion: number; base_rate: number; method: string } | null;
  provider: string;
  features: string[];
  validation: SimilarValidation | null;
  disclaimer: string;
}

export interface InterventionIn {
  player_id: string;
  /** -60..60 */
  x: number;
  /** -40..40 */
  y: number;
}

/**
 * Defender response policies. The backend accepts a fixed set; the lab reads the
 * available ones from GET /api/response-policies when the endpoint exists, so a
 * name outside this list is still carried through as a string.
 */
export type KnownResponsePolicy =
  | "none"
  | "marker_follows"
  | "press_carrier"
  | "zonal_shift"
  | "lane_cover"
  | "learned"
  | "learned_innovation";
export type ResponsePolicy = KnownResponsePolicy | (string & {});

/** Mirrors the policy description in ScenarioOut.policy and GET /api/response-policies. */
export interface PolicyInfo {
  name: ResponsePolicy;
  description: string;
  assumptions: string[];
  /** false = experimental: the policy has not been validated against observed defender movement */
  validated: boolean;
  /** where the validation lives (ledger claim, results file), or null */
  validation_ref: string | null;
  /**
   * The backend's verdict in words: "validated: ...", "validated (observational): ...",
   * "experimental: ..." or "no response simulated". Shown verbatim; the lab derives its badge
   * from it (src/lib/policies.ts policyBadge). Absent on older backends.
   */
  status?: string | null;
  /** machine-readable form of `status` (newer backends); the badge prefers it when present */
  verdict?: PolicyVerdict | null;
  /** for a linear, data-fitted response: the range of real movement it was fitted on (newer backends) */
  fitted_range?: FittedRangeOut | null;
}

export interface FittedRangeOut {
  /** share of real rows below limit_m, e.g. 0.99 */
  share: number;
  limit_m: number;
  horizon_s: number;
  quantity: string;
  /** committed results.json the limit is read from */
  ref: string;
}

export const POLICY_VERDICTS = [
  "no_response",
  "forecast_validated",
  "observational_validated",
  "observational_supported",
  "not_supported",
  "unchecked",
] as const;
export type PolicyVerdict = (typeof POLICY_VERDICTS)[number];
export type VelocityMode = "keep" | "zero";
export type ValueModel = "option_model" | "physics";

export interface ScenarioRequest {
  match_id: string;
  frame: number;
  /** 1..4 interventions */
  interventions: InterventionIn[];
  response_policy?: ResponsePolicy;
  follow_fraction?: number;
  velocity_mode?: VelocityMode;
  horizon_s?: number;
  ensemble?: number;
  seed?: number;
  /** which per-option probability the deltas use (default "option_model") */
  value_model?: ValueModel;
}

export interface FeasibilityOut {
  player_id: string;
  displacement_m: number;
  reachable_radius_m: number;
  reachable: boolean;
  origin_x: number | null;
  origin_y: number | null;
  inside_pitch: boolean;
  min_separation_m: number;
  supported: boolean;
  notes: string[];
  kind: "user" | "simulated";
  /**
   * Inside the kinematic envelope (reaction delay, acceleration and speed limits) from the observed
   * position and velocity horizon_s earlier; null when that origin is unobserved. Diagnostic only:
   * it does not change `supported`. Absent on older backends.
   */
  kinematic_reachable?: boolean | null;
}

export interface MovementOut {
  player_id: string;
  from_x: number;
  from_y: number;
  to_x: number;
  to_y: number;
  provenance: ProvenanceLabel;
  reason: string;
}

export interface OptionDeltaOut {
  receiver_id: string;
  observed_p: number | null;
  hypothetical_p: number | null;
  delta: number | null;
  observed_p10: number | null;
  observed_p90: number | null;
  hypothetical_p10: number | null;
  hypothetical_p90: number | null;
  /** share of the parameter ensemble whose delta has the same sign; null when |delta| < 0.005 (negligible) */
  delta_sign_agreement: number | null;
}

export type SummaryDict = Record<string, number | string | null>;

export interface ScenarioResponse {
  observed: FrameAnalysisOut;
  hypothetical: FrameAnalysisOut;
  feasibility: FeasibilityOut[];
  movements: MovementOut[];
  option_deltas: OptionDeltaOut[];
  summary_observed: SummaryDict;
  summary_hypothetical: SummaryDict;
  ensemble_size: number;
  supported: boolean;
  warnings: string[];
  assumptions: string[];
  disclaimer: string;
  /** which per-option probability option_deltas use */
  value_model: ValueModel;
  /** the defender response policy that produced the simulated movements (absent on older backends) */
  policy?: PolicyInfo | null;
}

/** One defender's experimental "defensive shadow" (mirrors DefenderShadowOut). */
export interface DefenderShadowOut {
  player_id: string;
  x: number;
  y: number;
  /** mean alt V_best - observed V_best (+ = suppressive) */
  shadow_best: number;
  /** mean alt V_sum - observed V_sum (+ = suppressive) */
  shadow_sum: number;
  alt_radius_m: number;
  /** nearby reachable positions the defender is compared against */
  alternatives: [number, number][];
  most_suppressed_receiver: string | null;
  most_suppressed_delta: number;
  /** |shadow_sum| above the tracking-noise floor */
  distinguishable_from_noise: boolean;
  provenance: ProvenanceLabel;
}

/**
 * GET /api/matches/{mid}/frames/{frame}/defensive-shadow (mirrors DefensiveShadowOut).
 * Responds 422 when there is no ball or no possession at the frame.
 */
export interface DefensiveShadowOut {
  match_id: string;
  frame: number;
  attacking_team: Team;
  carrier_id: string | null;
  value_model: ValueModel;
  v_best_observed: number;
  v_sum_observed: number;
  defenders: DefenderShadowOut[];
  assumptions: string[];
  /** |shadow_sum| below this is within tracking noise */
  noise_floor: number;
  horizon_s: number;
  alt_radius_m: number;
  status: string;
  /** provider the status refers to, the ledger claim behind it, and its committed results file (newer backends) */
  provider?: string | null;
  validation_claim?: string | null;
  validation_ref?: string | null;
}

/** Schema bounds for interventions, mirrored from InterventionIn. */
export const INTERVENTION_BOUNDS = { x: 60, y: 40 } as const;
export const MAX_INTERVENTIONS = 4;
