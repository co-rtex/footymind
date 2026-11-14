/**
 * Featured situations and the guided tour.
 *
 * The selection of each situation is documented in docs/RESEARCH.md ("Featured Possession
 * Multiverse example"). This file holds only the curated *identity* of each situation (match,
 * frame, the pass that was played, the defender examined and the move to try) and qualitative
 * story text. It holds no results: every number shown about a featured situation is fetched
 * from the API when it is displayed (see FeaturedFacts and the lab).
 */
import type {
  DefensiveShadowOut,
  MatchSummary,
  OptionDeltaOut,
  PlayerState,
  ScenarioResponse,
} from "../api/types";
import { fmtClock, matchClockSeconds } from "./format";
import { routeHash, type RouteMove } from "./hooks";
import type { TeamNames } from "./teams";

export type FeaturedId = "press" | "broadcast";

export interface FeaturedScenario {
  id: FeaturedId;
  matchId: string;
  frame: number;
  /** the hypothetical move the tour applies in step 3 (a user-style edit, labelled hypothetical) */
  moves: RouteMove[];
  /** the pass that was actually played at this frame (observed event) */
  passer: string;
  receiver: string;
  /** the defender whose position is examined and moved */
  defender: string;
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
    defender: "home_9",
    title: "One pressing position, one risky pass",
    source: "Metrica Sports · optical tracking",
    story: (t) =>
      `${t.away} build up from deep, near their own goal, under a high press. #17 plays to #18 and the pass is completed, but ${t.away} lose the ball moments later. One ${t.home} forward, #9, is standing where he takes away more passing value than any other defender.`,
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
    defender: "home_25",
    title: "The same question on broadcast data",
    source: "SkillCorner · broadcast tracking",
    story: (t) =>
      `${t.away} #27 passes to #14 and the pass is completed. ${t.home} #25 is the defender whose position takes away the most passing value. Broadcast cameras do not see every player, so positions away from the ball are provider extrapolations, drawn hollow and labelled as estimates.`,
    steps: (t) => ({
      watch: `Replay the real possession from broadcast tracking. Solid players are detected by the camera; hollow ones are off camera, and their positions are provider estimates, not observations.`,
      why: `Pause on the pass. See the space, the pressure on the ball and the passing options, and which inputs are estimates.`,
      move: `Move #25 ${t.home} within the region he could have reached and compare the estimates with what was observed.`,
    }),
  },
];

export function featuredById(id: string | null | undefined): FeaturedScenario | null {
  return FEATURED.find((f) => f.id === id) ?? null;
}

/** Deep link into the lab at the featured frame. With `tour`, the guided tour is shown; without, the move is applied. */
export function featuredHref(f: FeaturedScenario, mode: "tour" | "scenario"): string {
  return mode === "tour"
    ? routeHash({ name: "at", matchId: f.matchId, frame: f.frame, tour: f.id })
    : routeHash({ name: "at", matchId: f.matchId, frame: f.frame, moves: f.moves });
}

/** Facts about a featured situation, all read from API responses (null when not available). */
export interface FeaturedFacts {
  /** "P1 35:48", from the observed analysis and the match summary */
  clock: string | null;
  /** the pass that was played: observed vs hypothetical option-model estimate */
  pass: OptionDeltaOut | null;
  ensembleSize: number;
  /** settings (of ensembleSize) whose change has the same sign */
  agreeing: number | null;
  /** distance the defender is moved, and whether the edit passes the feasibility checks */
  displacement_m: number | null;
  supported: boolean;
  /** the examined defender's Defensive Shadow and the next highest among the other defenders */
  shadow: number | null;
  nextShadow: number | null;
  shadowIsTop: boolean | null;
  /** positions on the pitch at this frame, and how many are provider estimates */
  shown: number;
  estimated: number;
}

function onPitch(p: PlayerState): boolean {
  return p.x !== null && p.y !== null;
}

export function featuredFacts(
  f: FeaturedScenario,
  sc: ScenarioResponse,
  shadow: DefensiveShadowOut | null,
  match: MatchSummary | null,
): FeaturedFacts {
  const obs = sc.observed;
  const clock = match ? fmtClock(matchClockSeconds(obs.time_s, obs.period, match.periods, match.fps)) : null;
  const pass = sc.option_deltas.find((d) => d.receiver_id === f.receiver) ?? null;
  const agreeing =
    pass && pass.delta_sign_agreement !== null ? Math.round(pass.delta_sign_agreement * sc.ensemble_size) : null;
  const feas = sc.feasibility.find((q) => q.player_id === f.defender) ?? null;
  const shown = obs.players.filter(onPitch);
  let shadowVal: number | null = null;
  let next: number | null = null;
  let top: boolean | null = null;
  if (shadow) {
    const mine = shadow.defenders.find((d) => d.player_id === f.defender);
    const others = shadow.defenders.filter((d) => d.player_id !== f.defender).map((d) => d.shadow_sum);
    shadowVal = mine ? mine.shadow_sum : null;
    next = others.length ? Math.max(...others) : null;
    top = shadowVal !== null && next !== null ? shadowVal > next : null;
  }
  return {
    clock: clock ? `P${obs.period} ${clock}` : null,
    pass,
    ensembleSize: sc.ensemble_size,
    agreeing,
    displacement_m: feas ? feas.displacement_m : null,
    supported: sc.supported,
    shadow: shadowVal,
    nextShadow: next,
    shadowIsTop: top,
    shown: shown.length,
    estimated: shown.filter((p) => p.position_provenance === "estimated").length,
  };
}

