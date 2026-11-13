/**
 * Runtime checks and conversion of API payloads.
 *
 * The replay is converted to typed arrays once on arrival so the render loop
 * never touches boxed JSON numbers. `null` (not observed) becomes NaN in the
 * Float32Arrays, and every consumer must treat NaN as "do not draw".
 */
import type {
  DefensiveShadowOut,
  EventOut,
  FrameAnalysisOut,
  MatchSummary,
  Pitch,
  PlayerOut,
  PossessionSummary,
  ProvenanceLabel,
  Replay,
  ScenarioResponse,
  SourceInfo,
  Team,
} from "./types";

export class ContractError extends Error {
  constructor(message: string) {
    super(`API contract violation: ${message}`);
    this.name = "ContractError";
  }
}

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function need(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new ContractError(msg);
}

function needKeys(v: unknown, keys: string[], what: string): asserts v is Record<string, unknown> {
  need(isObj(v), `${what} is not an object`);
  for (const k of keys) need(k in v, `${what} is missing "${k}"`);
}

export function parseMatch(raw: unknown): MatchSummary {
  needKeys(raw, ["match_id", "source", "fps", "periods", "pitch"], "match");
  need(Array.isArray(raw.periods), "match.periods is not an array");
  return raw as unknown as MatchSummary;
}

export function parseMatches(raw: unknown): MatchSummary[] {
  need(Array.isArray(raw), "matches is not an array");
  for (const m of raw) parseMatch(m);
  return raw as MatchSummary[];
}

export function parsePossessions(raw: unknown): PossessionSummary[] {
  need(Array.isArray(raw), "possessions is not an array");
  for (const p of raw)
    needKeys(p, ["possession_id", "team", "period", "start_frame", "end_frame", "n_passes"], "possession");
  return raw as PossessionSummary[];
}

export function parseAnalysis(raw: unknown): FrameAnalysisOut {
  needKeys(raw, ["frame", "players", "pass_options", "attacking_team", "warnings", "model"], "analysis");
  need(Array.isArray(raw.players), "analysis.players is not an array");
  need(Array.isArray(raw.pass_options), "analysis.pass_options is not an array");
  const pc = raw.pitch_control;
  if (pc !== null && pc !== undefined) {
    needKeys(pc, ["nx", "ny", "x0", "y0", "dx", "dy", "values"], "pitch_control");
    need(Array.isArray(pc.values), "pitch_control.values is not an array");
    need(
      (pc.values as unknown[]).length === (pc.nx as number) * (pc.ny as number),
      `pitch_control.values has ${(pc.values as unknown[]).length} cells, expected nx*ny`,
    );
  }
  return raw as unknown as FrameAnalysisOut;
}

export function parseScenario(raw: unknown): ScenarioResponse {
  needKeys(
    raw,
    ["observed", "hypothetical", "feasibility", "movements", "option_deltas", "supported", "disclaimer"],
    "scenario",
  );
  parseAnalysis(raw.observed);
  parseAnalysis(raw.hypothetical);
  return raw as unknown as ScenarioResponse;
}

const SHADOW_DEFENDER_KEYS = [
  "player_id",
  "x",
  "y",
  "shadow_best",
  "shadow_sum",
  "alt_radius_m",
  "alternatives",
  "most_suppressed_receiver",
  "most_suppressed_delta",
  "distinguishable_from_noise",
];

export function parseDefensiveShadow(raw: unknown): DefensiveShadowOut {
  needKeys(raw, ["frame", "attacking_team", "defenders", "assumptions", "status", "v_sum_observed"], "defensive shadow");
  need(Array.isArray(raw.defenders), "defensive shadow defenders is not an array");
  need(Array.isArray(raw.assumptions), "defensive shadow assumptions is not an array");
  for (const d of raw.defenders as unknown[]) {
    needKeys(d, SHADOW_DEFENDER_KEYS, "shadow defender");
    need(typeof d.shadow_sum === "number" && Number.isFinite(d.shadow_sum), "shadow defender shadow_sum is not a number");
    need(typeof d.distinguishable_from_noise === "boolean", "shadow defender distinguishable_from_noise is not a boolean");
    need(Array.isArray(d.alternatives), "shadow defender alternatives is not an array");
    for (const a of d.alternatives as unknown[]) {
      need(
        Array.isArray(a) && a.length === 2 && a.every((v) => typeof v === "number"),
        "shadow alternative is not an [x, y] pair",
      );
    }
  }
  return raw as unknown as DefensiveShadowOut;
}

/** Replay in render-ready form. NaN marks unobserved values. */
export interface ReplayData {
  matchId: string;
  fps: number;
  step: number;
  pitch: Pitch;
  players: PlayerOut[];
  nPlayers: number;
  nFrames: number;
  frame: Int32Array;
  timeS: Float64Array;
  period: Int16Array;
  /** [frame][player][x|y], NaN when not observed */
  xy: Float32Array;
  /** [frame][x|y], NaN when not observed */
  ball: Float32Array;
  /**
   * [frame][player]: 1 = detected (observed), 0 = provider extrapolation
   * (estimated). null when the provider observes every drawn position.
   */
  detected: Uint8Array | null;
  /** number of non-null positions that are extrapolations (0 for optical tracking) */
  nEstimated: number;
  events: EventOut[];
  possession: PossessionSummary | null;
  /** period -> team -> +1 | -1 */
  attackSign: Map<number, Record<Team, number>>;
  provenance: ProvenanceLabel;
  source: SourceInfo;
}

function toF32(src: (number | null)[]): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    out[i] = typeof v === "number" && Number.isFinite(v) ? v : NaN;
  }
  return out;
}

export function parseReplay(raw: unknown): ReplayData {
  needKeys(
    raw,
    [
      "match_id",
      "fps",
      "step",
      "pitch",
      "players",
      "frame",
      "time_s",
      "period",
      "players_xy",
      "ball_xy",
      "events",
      "attack_sign",
      "source",
    ],
    "replay",
  );
  const r = raw as unknown as Replay;
  const nFrames = r.frame.length;
  const nPlayers = r.players.length;
  need(nFrames > 0, "replay has no frames");
  need(r.time_s.length === nFrames, "replay.time_s length differs from frame length");
  need(r.period.length === nFrames, "replay.period length differs from frame length");
  need(
    r.players_xy.length === nFrames * nPlayers * 2,
    `replay.players_xy has ${r.players_xy.length} values, expected ${nFrames * nPlayers * 2}`,
  );
  need(r.ball_xy.length === nFrames * 2, `replay.ball_xy has ${r.ball_xy.length} values, expected ${nFrames * 2}`);
  for (let i = 1; i < nFrames; i++) {
    need((r.frame[i] as number) > (r.frame[i - 1] as number), "replay.frame is not strictly increasing");
  }
  let detected: Uint8Array | null = null;
  let nEstimated = 0;
  const det = r.players_detected;
  if (det !== null && det !== undefined) {
    need(Array.isArray(det), "replay.players_detected is not an array");
    need(
      det.length === nFrames * nPlayers,
      `replay.players_detected has ${det.length} values, expected ${nFrames * nPlayers}`,
    );
    detected = new Uint8Array(det.length);
    for (let i = 0; i < det.length; i++) {
      const v = det[i];
      need(typeof v === "boolean", "replay.players_detected contains a non-boolean");
      detected[i] = v ? 1 : 0;
      if (!v && r.players_xy[2 * i] !== null && r.players_xy[2 * i] !== undefined) nEstimated++;
    }
  }
  const attackSign = new Map<number, Record<Team, number>>();
  for (const [p, signs] of Object.entries(r.attack_sign)) {
    attackSign.set(Number(p), { home: signs.home ?? 0, away: signs.away ?? 0 });
  }
  return {
    matchId: r.match_id,
    fps: r.fps,
    step: r.step,
    pitch: r.pitch,
    players: r.players,
    nPlayers,
    nFrames,
    frame: Int32Array.from(r.frame),
    timeS: Float64Array.from(r.time_s),
    period: Int16Array.from(r.period),
    xy: toF32(r.players_xy),
    ball: toF32(r.ball_xy),
    detected,
    nEstimated,
    events: r.events,
    possession: r.possession,
    attackSign,
    provenance: r.provenance,
    source: r.source,
  };
}
