/**
 * Featured situations and the guided tour.
 *
 * The selection of each situation is documented in docs/RESEARCH.md ("Featured Possession
 * Multiverse example" and "Featured response example"). This file holds only the curated
 * *identity* of each situation (match, frame, the pass that was played, the player examined and
 * moved, the move to try and, where relevant, the defender response to compare) and qualitative
 * story text. It holds no results: every number shown about a featured situation is fetched from
 * the API when it is displayed (see FeaturedFacts and the lab).
 */
import type {
  DefensiveShadowOut,
  MatchSummary,
  OptionDeltaOut,
  PlayerState,
  PolicyInfo,
  ResponsePolicy,
  ScenarioResponse,
} from "../api/types";
import { isNegligible } from "./agreement";
import { fmtClock, matchClockSeconds } from "./format";
import { routeHash, type RouteMove } from "./hooks";
import type { TeamNames } from "./teams";

export type FeaturedId = "press" | "broadcast" | "response";

export interface FeaturedScenario {
  id: FeaturedId;
  matchId: string;
  frame: number;
  /** the hypothetical move the tour applies in step 3 (a user-style edit, labelled hypothetical) */
  moves: RouteMove[];
  /** the pass that was actually played at this frame (observed event) */
  passer: string;
  receiver: string;
  /** the player whose position is examined and moved (a defender or an attacker) */
  moved: string;
  /** show the moved defender's Defensive Shadow (experimental) on the card */
  showShadow: boolean;
  /**
   * A defender response to compare with static defenders: the card shows both results, and tour
   * step 3 and the direct what-if link preselect it, so the lab shows its static-vs-response
   * comparison. Absent: static defenders only.
   */
  response?: ResponsePolicy;
  title: string;
  /** short source line, e.g. "Metrica Sports · optical tracking" */
  source: string;
  /** qualitative story; team display names are filled in (never numbers) */
  story: (t: TeamNames) => string;
  /** one sentence per tour step */
  steps: (t: TeamNames) => { watch: string; why: string; move: string };
}

export const FEATURED: readonly FeaturedScenario[] = [
  {
    id: "press",
    matchId: "metrica-sample-1",
    frame: 53708,
    moves: [{ player_id: "home_9", x: 50.7, y: -8.75 }],
    passer: "away_17",
    receiver: "away_18",
    moved: "home_9",
    showShadow: true,
    title: "One pressing position, one risky pass",
    source: "Metrica Sports · optical tracking",
    story: (t) =>
      `${t.away} build up from deep, near their own goal, under a high press. #17 plays to #18 and the pass is completed, but ${t.away} lose the ball moments later. By the experimental Defensive Shadow model, one ${t.home} forward, #9, is standing where his position takes away the most passing value; that model is not validated as identifying who stops passes.`,
    steps: (t) => ({
      watch: `Replay the real possession: ${t.away} play out from the back while ${t.home} press high. Every player and ball position drawn in the replay is observed tracking data.`,
      why: `Pause on the pass. The pitch shows the space each team controls, the passing options and the pressure on the ball; the panel says what stands out and why.`,
      move: `Move #9 ${t.home} a few metres, within the region he could have reached, and compare the model's estimates with what was observed.`,
    }),
  },
  {
    id: "broadcast",
    matchId: "skillcorner-1899585",
    frame: 39177,
    moves: [{ player_id: "home_25", x: 22.46, y: -5.77 }],
    passer: "away_27",
    receiver: "away_14",
    moved: "home_25",
    showShadow: true,
    title: "The same question on broadcast data",
    source: "SkillCorner · broadcast tracking",
    story: (t) =>
      `${t.away} #27 passes to #14 and the pass is completed. By the experimental Defensive Shadow model, ${t.home} #25 is the defender whose position takes away the most passing value. Broadcast cameras do not see every player, so positions away from the ball are provider extrapolations, drawn hollow and labelled as estimates.`,
    steps: (t) => ({
      watch: `Replay the real possession from broadcast tracking. Solid players are detected by the camera; hollow ones are off camera, and their positions are provider estimates, not observations.`,
      why: `Pause on the pass. See the space, the pressure on the ball and the passing options, and which inputs are estimates.`,
      move: `Move #25 ${t.home} within the region he could have reached and compare the estimates with what was observed.`,
    }),
  },
  {
    id: "response",
    matchId: "metrica-sample-2",
    frame: 31360,
    moves: [{ player_id: "home_5", x: -29.04, y: 9.47 }],
    passer: "home_1",
    receiver: "home_5",
    moved: "home_5",
    showShadow: false,
    response: "marker_follows",
    title: "One run, two assumptions about the defence",
    source: "Metrica Sports · optical tracking",
    story: (t) =>
      `${t.home} #1 passes to #5 and the pass is completed. Suppose #5 had first run a few metres into space. If the defenders stand still, the option model rates the pass as more likely to arrive. If his marker follows him, as markers did in an observational check of the tracking data, the modelled gain disappears. The conclusion rests on an assumption about the defence: this is a model-based illustration, not what happened in the match.`,
    steps: (t) => ({
      watch: `Replay the real possession: ${t.home} #1 plays the ball to #5. Every player and ball position drawn in the replay is observed tracking data.`,
      why: `Pause on the pass. The pitch shows the space each team controls, the passing options and the pressure on the ball. Note the ${t.away} defender close to #5.`,
      move: `Move #5 ${t.home} a few metres into space, then compare two assumptions: defenders who stand still, and a marker who follows him (a simulated move). Both results are model estimates.`,
    }),
  },
];

/**
 * The featured situations whose match this server has (GET /api/matches), in curated order. A
 * Metrica-only setup (the README quick start, the Docker default) has no SkillCorner bundles, so
 * the broadcast situation is left out instead of shown as a broken card.
 */
export function availableFeatured(
  matchIds: Iterable<string>,
  list: readonly FeaturedScenario[] = FEATURED,
): FeaturedScenario[] {
  const ids = new Set(matchIds);
  return list.filter((f) => ids.has(f.matchId));
}

export function featuredById(id: string | null | undefined): FeaturedScenario | null {
  return FEATURED.find((f) => f.id === id) ?? null;
}

/**
 * Deep link into the lab at the featured frame. With `tour`, the guided tour is shown; without,
 * the move is applied (with the situation's defender response preselected, if it has one).
 */
export function featuredHref(f: FeaturedScenario, mode: "tour" | "scenario"): string {
  return mode === "tour"
    ? routeHash({ name: "at", matchId: f.matchId, frame: f.frame, tour: f.id })
    : routeHash({ name: "at", matchId: f.matchId, frame: f.frame, moves: f.moves, response: f.response });
}

/** The moved defender's Defensive Shadow (experimental) and the next highest among the other defenders. */
export interface ShadowFact {
  value: number;
  next: number | null;
  /** the moved defender has the highest shadow; null when there is no other defender to compare */
  isTop: boolean | null;
}

/** A defender the response policy moved: simulated, never observed. */
export interface SimulatedMoveFact {
  player_id: string;
  distance_m: number;
}

/** The same edit under the situation's defender response, read from that scenario response. */
export interface ResponseFact {
  policy: ResponsePolicy;
  /** the policy's description and validation status, as the scenario response states them */
  info: PolicyInfo | null;
  /** the pass that was played, observed vs hypothetical, under the response */
  pass: OptionDeltaOut | null;
  agreeing: number | null;
  ensembleSize: number;
  supported: boolean;
  simulated: SimulatedMoveFact[];
  /**
   * The change in the played pass points the other way than with static defenders. Null when
   * either change is missing or negligible (then its direction is not meaningful).
   */
  flips: boolean | null;
}

/** Facts about a featured situation, all read from API responses (null when not available). */
export interface FeaturedFacts {
  /** "P1 35:48", from the observed analysis and the match summary */
  clock: string | null;
  /** the pass that was played: observed vs hypothetical option-model estimate, defenders static */
  pass: OptionDeltaOut | null;
  ensembleSize: number;
  /** settings (of ensembleSize) whose change has the same sign */
  agreeing: number | null;
  /** distance the player is moved, and whether the edit passes the feasibility checks */
  displacement_m: number | null;
  supported: boolean;
  /** situations that show it, when the shadow response includes the moved defender */
  shadow: ShadowFact | null;
  /** situations with a defender response, once that result is available */
  response: ResponseFact | null;
  /** positions on the pitch at this frame, and how many are provider estimates */
  shown: number;
  estimated: number;
}

/** The API responses the facts are read from. */
export interface FeaturedPayloads {
  /** the featured move with static defenders */
  scenario: ScenarioResponse;
  /** the Defensive Shadow at the featured frame */
  shadow?: DefensiveShadowOut | null;
  /** the same move under the situation's response policy */
  response?: ScenarioResponse | null;
  match?: MatchSummary | null;
}

function playedPass(f: FeaturedScenario, sc: ScenarioResponse): { pass: OptionDeltaOut | null; agreeing: number | null } {
  const pass = sc.option_deltas.find((d) => d.receiver_id === f.receiver) ?? null;
  const agreeing =
    pass && pass.delta_sign_agreement !== null ? Math.round(pass.delta_sign_agreement * sc.ensemble_size) : null;
  return { pass, agreeing };
}

function onPitch(p: PlayerState): boolean {
  return p.x !== null && p.y !== null;
}

function shadowFact(f: FeaturedScenario, shadow: DefensiveShadowOut | null | undefined): ShadowFact | null {
  if (!f.showShadow || !shadow) return null;
  const mine = shadow.defenders.find((d) => d.player_id === f.moved);
  if (!mine) return null;
  const others = shadow.defenders.filter((d) => d.player_id !== f.moved).map((d) => d.shadow_sum);
  const next = others.length ? Math.max(...others) : null;
  return { value: mine.shadow_sum, next, isTop: next !== null ? mine.shadow_sum > next : null };
}

function responseFact(
  f: FeaturedScenario,
  staticPass: OptionDeltaOut | null,
  sc: ScenarioResponse | null | undefined,
): ResponseFact | null {
  if (!f.response || !sc) return null;
  const { pass, agreeing } = playedPass(f, sc);
  const simulated = sc.movements
    .filter((m) => m.provenance === "simulated")
    .map((m) => ({ player_id: m.player_id, distance_m: Math.hypot(m.to_x - m.from_x, m.to_y - m.from_y) }));
  const a = staticPass?.delta ?? null;
  const b = pass?.delta ?? null;
  const flips = a === null || b === null || isNegligible(a) || isNegligible(b) ? null : Math.sign(a) !== Math.sign(b);
  return {
    policy: f.response,
    info: sc.policy && sc.policy.name === f.response ? sc.policy : null,
    pass,
    agreeing,
    ensembleSize: sc.ensemble_size,
    supported: sc.supported,
    simulated,
    flips,
  };
}

export function featuredFacts(f: FeaturedScenario, { scenario: sc, shadow, response, match }: FeaturedPayloads): FeaturedFacts {
  const obs = sc.observed;
  const clock = match ? fmtClock(matchClockSeconds(obs.time_s, obs.period, match.periods, match.fps)) : null;
  const { pass, agreeing } = playedPass(f, sc);
  const feas = sc.feasibility.find((q) => q.player_id === f.moved) ?? null;
  const shown = obs.players.filter(onPitch);
  return {
    clock: clock ? `P${obs.period} ${clock}` : null,
    pass,
    ensembleSize: sc.ensemble_size,
    agreeing,
    displacement_m: feas ? feas.displacement_m : null,
    supported: sc.supported,
    shadow: shadowFact(f, shadow),
    response: responseFact(f, pass, response),
    shown: shown.length,
    estimated: shown.filter((p) => p.position_provenance === "estimated").length,
  };
}
