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
  PolicyInfo,
  PolicyVerdict,
  PossessionSummary,
  ProvenanceLabel,
  Replay,
  ScenarioResponse,
  SimilarMomentsOut,
  SourceInfo,
  StatementOut,
  Team,
} from "./types";
import { POLICY_VERDICTS } from "./types";

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

function isStrOrNull(v: unknown): boolean {
  return v === undefined || v === null || typeof v === "string";
}

export function parseMatch(raw: unknown): MatchSummary {
  needKeys(raw, ["match_id", "source", "fps", "periods", "pitch"], "match");
  need(Array.isArray(raw.periods), "match.periods is not an array");
  for (const k of ["home_label", "away_label", "competition", "date"]) {
    need(isStrOrNull(raw[k]), `match.${k} is not a string or null`);
  }
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

function isNumOrNull(v: unknown): boolean {
  return v === null || (typeof v === "number" && Number.isFinite(v));
}

export function parseStatement(raw: unknown, what = "statement"): StatementOut {
  needKeys(raw, ["kind", "text", "metric", "provenance", "players"], what);
  need(typeof raw.kind === "string", `${what}.kind is not a string`);
  need(typeof raw.text === "string" && raw.text.length > 0, `${what}.text is not a non-empty string`);
  need(typeof raw.metric === "string", `${what}.metric is not a string`);
  need(typeof raw.provenance === "string", `${what}.provenance is not a string`);
  need(Array.isArray(raw.players) && raw.players.every((p) => typeof p === "string"), `${what}.players is not an array of ids`);
  const v = raw.value;
  need(v === undefined || v === null || typeof v === "string" || typeof v === "number", `${what}.value has an invalid type`);
  return raw as unknown as StatementOut;
}

export function parseAnalysis(raw: unknown): FrameAnalysisOut {
  needKeys(raw, ["frame", "players", "pass_options", "attacking_team", "warnings", "model"], "analysis");
  // Optional (newer backends): causal temporal features and explanation statements.
  const t = raw.temporal;
  if (t !== undefined && t !== null) {
    need(isObj(t), "analysis.temporal is not an object");
    for (const [k, v] of Object.entries(t)) need(isNumOrNull(v), `analysis.temporal.${k} is not a number or null`);
  }
  const ex = raw.explanations;
  if (ex !== undefined && ex !== null) {
    need(Array.isArray(ex), "analysis.explanations is not an array");
    ex.forEach((s, i) => parseStatement(s, `analysis.explanations[${i}]`));
  }
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

/**
 * One response-policy description. `validation_ref` and `status` may be absent (treated as
 * null); every other field is required when a policy object is present.
 */
export function parsePolicyInfo(raw: unknown, what = "policy"): PolicyInfo {
  needKeys(raw, ["name", "description", "assumptions", "validated"], what);
  need(typeof raw.name === "string" && raw.name.length > 0, `${what}.name is not a non-empty string`);
  need(typeof raw.description === "string", `${what}.description is not a string`);
  need(
    Array.isArray(raw.assumptions) && raw.assumptions.every((a) => typeof a === "string"),
    `${what}.assumptions is not an array of strings`,
  );
  need(typeof raw.validated === "boolean", `${what}.validated is not a boolean`);
  const ref = raw.validation_ref;
  need(ref === undefined || ref === null || typeof ref === "string", `${what}.validation_ref is not a string or null`);
  // The verdict text (newer backends). Absent or null on older backends: the badge falls back.
  const status = raw.status;
  need(status === undefined || status === null || typeof status === "string", `${what}.status is not a string or null`);
  // Machine-readable verdict (newer backends). An unknown value is treated as absent so a newer
  // backend's vocabulary does not break the lab; the badge then falls back to the status text.
  const verdict = raw.verdict;
  need(verdict === undefined || verdict === null || typeof verdict === "string", `${what}.verdict is not a string or null`);
  return {
    name: raw.name,
    description: raw.description,
    assumptions: raw.assumptions as string[],
    validated: raw.validated,
    validation_ref: typeof ref === "string" ? ref : null,
    status: typeof status === "string" && status.trim() ? status.trim() : null,
    verdict: (POLICY_VERDICTS as readonly string[]).includes(verdict as string) ? (verdict as PolicyVerdict) : null,
  };
}

export function parseResponsePolicies(raw: unknown): PolicyInfo[] {
  need(Array.isArray(raw), "response policies is not an array");
  const out = raw.map((p, i) => parsePolicyInfo(p, `response policy ${i}`));
  need(new Set(out.map((p) => p.name)).size === out.length, "response policies contain duplicate names");
  return out;
}

const MOVEMENT_KEYS = ["player_id", "from_x", "from_y", "to_x", "to_y", "provenance"];

export function parseScenario(raw: unknown): ScenarioResponse {
  needKeys(
    raw,
    ["observed", "hypothetical", "feasibility", "movements", "option_deltas", "supported", "disclaimer"],
    "scenario",
  );
  parseAnalysis(raw.observed);
  parseAnalysis(raw.hypothetical);
  need(Array.isArray(raw.movements), "scenario.movements is not an array");
  for (const m of raw.movements as unknown[]) {
    needKeys(m, MOVEMENT_KEYS, "scenario movement");
    for (const k of ["from_x", "from_y", "to_x", "to_y"]) {
      need(typeof m[k] === "number" && Number.isFinite(m[k]), `scenario movement ${k} is not a number`);
    }
  }
  need(Array.isArray(raw.option_deltas), "scenario.option_deltas is not an array");
  // `policy` is optional (older backends do not send it); when present it must be well formed.
  if (raw.policy !== undefined && raw.policy !== null) {
    const policy = parsePolicyInfo(raw.policy, "scenario.policy");
    return { ...(raw as unknown as ScenarioResponse), policy };
  }
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
  /** [frame]: 1 = ball detected (observed), 0 = provider extrapolation (estimated); null when not provided */
  ballDetected: Uint8Array | null;
  /** number of frames whose non-null ball position is an extrapolation */
  nBallEstimated: number;
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
  let ballDetected: Uint8Array | null = null;
  let nBallEstimated = 0;
  const bdet = r.ball_detected;
  if (bdet !== null && bdet !== undefined) {
    need(Array.isArray(bdet), "replay.ball_detected is not an array");
    need(bdet.length === nFrames, `replay.ball_detected has ${bdet.length} values, expected ${nFrames}`);
    ballDetected = new Uint8Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
      const v = bdet[i];
      need(typeof v === "boolean", "replay.ball_detected contains a non-boolean");
      ballDetected[i] = v ? 1 : 0;
      if (!v && r.ball_xy[2 * i] !== null && r.ball_xy[2 * i] !== undefined) nBallEstimated++;
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
    ballDetected,
    nBallEstimated,
    events: r.events,
    possession: r.possession,
    attackSign,
    provenance: r.provenance,
    source: r.source,
  };
}

const NEIGHBOUR_KEYS = ["match_id", "frame", "team", "passer_id", "completed", "distance"];

export function parseSimilar(raw: unknown): SimilarMomentsOut {
  needKeys(
    raw,
    ["match_id", "frame", "decision_frame", "passer_id", "k", "neighbours", "completed_share", "provider", "disclaimer"],
    "similar moments",
  );
  need(Array.isArray(raw.neighbours), "similar moments neighbours is not an array");
  (raw.neighbours as unknown[]).forEach((n, i) => {
    const what = `similar neighbour ${i}`;
    needKeys(n, NEIGHBOUR_KEYS, what);
    need(typeof n.completed === "boolean", `${what}.completed is not a boolean`);
    need(typeof n.distance === "number" && Number.isFinite(n.distance), `${what}.distance is not a number`);
    need(Number.isInteger(n.frame), `${what}.frame is not an integer`);
    need(n.period === undefined || n.period === null || Number.isInteger(n.period), `${what}.period is not an integer`);
    for (const k of ["ball_x_att", "press_nearest_def_m", "passer_control_s", "match_minute"]) {
      need(n[k] === undefined || isNumOrNull(n[k]), `${what}.${k} is not a number or null`);
    }
  });
  need(typeof raw.disclaimer === "string" && raw.disclaimer.length > 0, "similar moments disclaimer is missing");
  const val = raw.validation;
  let validation: SimilarMomentsOut["validation"] = null;
  if (val !== undefined && val !== null) {
    need(isObj(val), "similar moments validation is not an object");
    const kb = val.knn_beats_base_rate;
    need(kb === undefined || kb === null || typeof kb === "boolean", "validation.knn_beats_base_rate is not a boolean or null");
    const ref = val.ref;
    need(ref === undefined || ref === null || typeof ref === "string", "validation.ref is not a string or null");
    validation = {
      ref: typeof ref === "string" ? ref : null,
      knn_beats_base_rate: typeof kb === "boolean" ? kb : null,
      log_loss: val.log_loss ?? null,
    };
  }
  const features = Array.isArray(raw.features) ? (raw.features as string[]) : [];
  return { ...(raw as unknown as SimilarMomentsOut), features, validation };
}
