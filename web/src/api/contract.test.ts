/**
 * Key parity between src/api/types.ts and what the real backend sends.
 *
 * src/test/api/keys.json holds the key NAMES (no values) of real responses and
 * response-policies.json the real GET /api/response-policies payload (text only, no tracking
 * data), both captured by web/scripts/capture-api-samples.mjs. Re-run it after a schema change.
 */
import { describe, expect, it } from "vitest";
import sample from "../test/api/keys.json";
import realPolicies from "../test/api/response-policies.json";
import { POLICY_BADGE_TEXT, policyBadge, policyFitRange } from "../lib/policies";
import { parseResponsePolicies } from "./parse";
import type {
  DefenderShadowOut,
  DefensiveShadowOut,
  EventOut,
  FeasibilityOut,
  FrameAnalysisOut,
  MatchSummary,
  ModelInfo,
  MovementOut,
  OptionDeltaOut,
  PassOptionOut,
  PeriodInfo,
  PitchControlOut,
  PlayerOut,
  PlayerState,
  PolicyInfo,
  PossessionSummary,
  PressureOut,
  Replay,
  ScenarioResponse,
  SimilarMoment,
  SimilarMomentsOut,
  SourceInfo,
  StatementOut,
} from "./types";

/**
 * Every key of T, enforced by the compiler: a key missing from the literal is a type error, and
 * so is a key T does not have (excess property check). So the lists below are types.ts itself.
 */
function keysOf<T>(r: Record<keyof T, true>): string[] {
  return Object.keys(r).sort();
}

const TS: Record<string, string[]> = {
  MatchSummary: keysOf<MatchSummary>({
    match_id: true, source: true, fps: true, n_frames: true, periods: true, n_possessions: true, n_events: true,
    pitch: true, home_label: true, away_label: true, competition: true, date: true,
  }),
  SourceInfo: keysOf<SourceInfo>({ provider: true, name: true, url: true, license_note: true }),
  PeriodInfo: keysOf<PeriodInfo>({
    period: true, first_frame: true, last_frame: true, duration_s: true, home_attacks_positive_x: true,
  }),
  PossessionSummary: keysOf<PossessionSummary>({
    possession_id: true, team: true, period: true, start_frame: true, end_frame: true, start_time_s: true,
    duration_s: true, n_events: true, n_passes: true, first_type: true, first_subtype: true, last_type: true,
    last_subtype: true, has_shot: true, has_goal: true,
  }),
  Replay: keysOf<Replay>({
    match_id: true, fps: true, step: true, pitch: true, players: true, frame: true, time_s: true, period_time_s: true,
    period: true, players_xy: true, players_detected: true, ball_xy: true, ball_detected: true, events: true,
    possession: true, attack_sign: true, provenance: true, source: true,
  }),
  PlayerOut: keysOf<PlayerOut>({ player_id: true, team: true, jersey: true }),
  EventOut: keysOf<EventOut>({
    event_id: true, team: true, type: true, subtype: true, period: true, start_frame: true, end_frame: true,
    start_time_s: true, from_player_id: true, to_player_id: true, start_x: true, start_y: true, end_x: true,
    end_y: true, possession_id: true,
  }),
  FrameAnalysisOut: keysOf<FrameAnalysisOut>({
    match_id: true, frame: true, period: true, time_s: true, period_time_s: true, reach_horizon_s: true,
    reach_speed_mps: true, attacking_team: true,
    attack_sign: true, possession_id: true, carrier_id: true, ball_x: true, ball_y: true, ball_provenance: true,
    players: true, pitch_control: true, pass_options: true, pressure: true, control_area_m2: true,
    control_area_final_third_m2: true, model: true, option_model: true, warnings: true, temporal: true,
    explanations: true,
  }),
  PlayerState: keysOf<PlayerState>({
    player_id: true, team: true, jersey: true, x: true, y: true, vx: true, vy: true, is_gk: true,
    position_provenance: true, velocity_provenance: true, reach_origin_x: true, reach_origin_y: true,
    reach_radius_m: true,
  }),
  PitchControlOut: keysOf<PitchControlOut>({
    nx: true, ny: true, x0: true, y0: true, dx: true, dy: true, values: true, provenance: true,
  }),
  PassOptionOut: keysOf<PassOptionOut>({
    receiver_id: true, target_x: true, target_y: true, distance_m: true, forward_progress_m: true,
    ball_travel_time_s: true, p_control_at_target: true, p_lane_clear: true, p_success: true, p_success_model: true,
    lane_min_margin_s: true, lane_closest_defender_id: true, lane_closest_defender_perp_m: true,
    defenders_within_3m_of_lane: true, provenance: true,
  }),
  PressureOut: keysOf<PressureOut>({
    nearest_defender_id: true, nearest_defender_distance_m: true, closing_speed_mps: true, defenders_within_5m: true,
    defenders_within_10m: true, provenance: true,
  }),
  ModelInfo: keysOf<ModelInfo>({ name: true, reference: true, status: true, params: true, provider: true, resource: true }),
  StatementOut: keysOf<StatementOut>({ kind: true, text: true, metric: true, value: true, provenance: true, players: true }),
  PolicyInfo: keysOf<PolicyInfo>({
    name: true, description: true, assumptions: true, validated: true, validation_ref: true, status: true, verdict: true,
    fitted_range: true,
  }),
  ScenarioResponse: keysOf<ScenarioResponse>({
    observed: true, hypothetical: true, feasibility: true, movements: true, option_deltas: true,
    summary_observed: true, summary_hypothetical: true, ensemble_size: true, supported: true, warnings: true,
    assumptions: true, disclaimer: true, value_model: true, policy: true,
  }),
  FeasibilityOut: keysOf<FeasibilityOut>({
    player_id: true, displacement_m: true, reachable_radius_m: true, reachable: true, origin_x: true, origin_y: true,
    inside_pitch: true, min_separation_m: true, supported: true, notes: true, kind: true, kinematic_reachable: true,
  }),
  MovementOut: keysOf<MovementOut>({
    player_id: true, from_x: true, from_y: true, to_x: true, to_y: true, provenance: true, reason: true,
  }),
  OptionDeltaOut: keysOf<OptionDeltaOut>({
    receiver_id: true, observed_p: true, hypothetical_p: true, delta: true, observed_p10: true, observed_p90: true,
    hypothetical_p10: true, hypothetical_p90: true, delta_sign_agreement: true,
  }),
  DefensiveShadowOut: keysOf<DefensiveShadowOut>({
    match_id: true, frame: true, attacking_team: true, carrier_id: true, value_model: true, v_best_observed: true,
    v_sum_observed: true, defenders: true, assumptions: true, noise_floor: true, horizon_s: true, alt_radius_m: true,
    status: true, provider: true, validation_claim: true, validation_ref: true,
  }),
  DefenderShadowOut: keysOf<DefenderShadowOut>({
    player_id: true, x: true, y: true, shadow_best: true, shadow_sum: true, alt_radius_m: true, alternatives: true,
    most_suppressed_receiver: true, most_suppressed_delta: true, distinguishable_from_noise: true, provenance: true,
  }),
  SimilarMomentsOut: keysOf<SimilarMomentsOut>({
    match_id: true, frame: true, decision_frame: true, passer_id: true, k: true, neighbours: true,
    completed_share: true, estimate: true, provider: true, features: true, validation: true, disclaimer: true,
  }),
  SimilarMoment: keysOf<SimilarMoment>({
    match_id: true, frame: true, possession_id: true, team: true, passer_id: true, completed: true, distance: true,
    ball_x_att: true, press_nearest_def_m: true, passer_control_s: true, period: true, match_minute: true,
  }),
};
TS.OptionModelInfo = TS.ModelInfo!;

const real = sample as unknown as Record<string, string[]>;

describe("types.ts mirrors the keys of real API responses", () => {
  const names = Object.keys(real).filter((k) => Array.isArray(real[k]));

  it("covers every captured response object", () => {
    expect(names.length).toBeGreaterThan(20);
    expect(names.filter((n) => !(n in TS))).toEqual([]);
  });

  it.each(names)("%s: same keys as the backend sends", (name) => {
    expect(TS[name]).toEqual([...real[name]!].sort());
  });

  it("the physics model sends provider and resource as null, which ModelInfo allows", () => {
    const physics: Pick<ModelInfo, "provider" | "resource"> = sample.physicsModel;
    expect(physics).toEqual({ provider: null, resource: null });
  });
});

describe("policy verdicts from the real GET /api/response-policies payload", () => {
  const list = parseResponsePolicies(realPolicies);
  const byName = new Map(list.map((p) => [p.name, p]));

  it("keeps the status text the backend sends", () => {
    for (const raw of realPolicies) expect(byName.get(raw.name)?.status).toBe(raw.status);
  });

  it("derives one badge vocabulary from status and validated", () => {
    const badges = Object.fromEntries(list.map((p) => [p.name, policyBadge(p)]));
    expect(badges).toEqual({
      none: null,
      marker_follows: "observational",
      lane_cover: "observational",
      learned_innovation: "observational",
      learned: "forecaster",
      press_carrier: "not supported",
      zonal_shift: "not supported",
    });
    for (const b of Object.values(badges)) if (b) expect(POLICY_BADGE_TEXT[b]).toBeTruthy();
  });

  it("reads learned_innovation's fitted range from its served fitted_range, and no other policy has one", () => {
    const served = realPolicies.find((p) => p.name === "learned_innovation")?.fitted_range;
    if (!served) throw new Error("the captured payload has no fitted_range for learned_innovation");
    expect(policyFitRange(byName.get("learned_innovation"))).toEqual({
      sharePct: Math.round(served.share * 1000) / 10,
      limitM: Math.round(served.limit_m * 10) / 10,
      horizonS: served.horizon_s,
    });
    for (const p of list) if (p.name !== "learned_innovation") expect(policyFitRange(p)).toBeNull();
  });
});
