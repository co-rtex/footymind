#!/usr/bin/env node
/**
 * Refresh the small API samples used by the contract tests (src/test/api/):
 *   - response-policies.json: GET /api/response-policies as served (text only, no tracking data)
 *   - keys.json: the key NAMES of each response object, taken from real responses (no values)
 *
 * Needs a running backend with metrica-sample-1 ingested:
 *   FOOTYMIND_API=http://127.0.0.1:8000 node scripts/capture-api-samples.mjs
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const API = (process.env.FOOTYMIND_API ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "test", "api");
const MATCH = "metrica-sample-1";
const FRAME = 53708;

async function get(path, init) {
  const res = await fetch(`${API}/api${path}`, init);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

const keys = (o) => Object.keys(o).sort();

const matches = await get("/matches");
const match = matches.find((m) => m.match_id === MATCH);
if (!match) throw new Error(`${MATCH} is not ingested on ${API}`);
const possessions = await get(`/matches/${MATCH}/possessions?min_passes=3`);
const possession = await get(`/matches/${MATCH}/frames/${FRAME}/possession`);
const replay = await get(`/possessions/${encodeURIComponent(possession.possession_id)}/replay`);
const analysis = await get(`/matches/${MATCH}/frames/${FRAME}/analysis`);
const policies = await get(`/response-policies?match_id=${MATCH}`);
const shadow = await get(`/matches/${MATCH}/frames/${FRAME}/defensive-shadow?horizon_s=1.0&value_model=option_model`);
const similar = await get(`/matches/${MATCH}/frames/${FRAME}/similar?k=8`);
const attacker = analysis.players.find(
  (p) => p.team === analysis.attacking_team && p.x !== null && !p.is_gk && p.player_id !== analysis.carrier_id,
);
const scenario = await get("/scenarios", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    match_id: MATCH,
    frame: FRAME,
    interventions: [{ player_id: attacker.player_id, x: attacker.x - 2, y: attacker.y }],
    response_policy: "marker_follows",
    horizon_s: 1.5,
    ensemble: 4,
  }),
});

const out = {
  _note: "Key names of real API responses (no values), captured by web/scripts/capture-api-samples.mjs.",
  MatchSummary: keys(match),
  SourceInfo: keys(match.source),
  PeriodInfo: keys(match.periods[0]),
  PossessionSummary: keys(possessions[0]),
  Replay: keys(replay),
  PlayerOut: keys(replay.players[0]),
  EventOut: keys(replay.events[0]),
  FrameAnalysisOut: keys(analysis),
  PlayerState: keys(analysis.players[0]),
  PitchControlOut: keys(analysis.pitch_control),
  PassOptionOut: keys(analysis.pass_options[0]),
  PressureOut: keys(analysis.pressure),
  ModelInfo: keys(analysis.model),
  OptionModelInfo: keys(analysis.option_model),
  StatementOut: keys(analysis.explanations[0]),
  PolicyInfo: keys(policies[0]),
  ScenarioResponse: keys(scenario),
  FeasibilityOut: keys(scenario.feasibility[0]),
  MovementOut: keys(scenario.movements[0]),
  OptionDeltaOut: keys(scenario.option_deltas[0]),
  DefensiveShadowOut: keys(shadow),
  DefenderShadowOut: keys(shadow.defenders[0]),
  SimilarMomentsOut: keys(similar),
  SimilarMoment: keys(similar.neighbours[0]),
  // the physics model sends these keys as null (ModelInfo.provider / resource are string | null)
  physicsModel: { provider: analysis.model.provider, resource: analysis.model.resource },
};

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, "keys.json"), `${JSON.stringify(out, null, 2)}\n`);
writeFileSync(join(OUT, "response-policies.json"), `${JSON.stringify(policies, null, 2)}\n`);
console.log(`wrote ${OUT}/keys.json and response-policies.json from ${API}`);
