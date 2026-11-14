/**
 * First-visit journey: zones, featured facts, routes, the guided tour, the player card, the
 * featured card and the response hint. Every payload here is SYNTHETIC (made-up numbers that
 * only exercise the code paths; none of them is a result).
 */
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type {
  DefenderShadowOut,
  DefensiveShadowOut,
  FeasibilityOut,
  MatchSummary,
  PlayerOut,
  PlayerState,
  ScenarioResponse,
} from "../api/types";
import { FEATURED, featuredById, featuredFacts, featuredHref } from "../lib/featured";
import { resetFeaturedLive } from "../lib/featuredLive";
import { parseRoute, routeHash } from "../lib/hooks";
import { describeZone, pitchZone } from "../lib/zones";
import { SYNTHETIC_SOURCE, syntheticAnalysis } from "../test/fixtures";
import { FeaturedCard } from "./FeaturedCard";
import { GuideBar } from "./GuideBar";
import { PlayerInspector } from "./PlayerInspector";
import { ProvenanceKey } from "./ProvenanceKey";
import { ScenarioPanel } from "./ScenarioPanel";

const PITCH = { length: 105, width: 68 };
const PRESS = FEATURED[0]!;

function state(id: string, team: "home" | "away", jersey: number, x: number | null, prov: PlayerState["position_provenance"] = "observed"): PlayerState {
  return {
    player_id: id,
    team,
    jersey,
    x,
    y: x === null ? null : 0,
    vx: 0,
    vy: 0,
    is_gk: false,
    position_provenance: prov,
    velocity_provenance: "derived",
    reach_origin_x: null,
    reach_origin_y: null,
    reach_radius_m: null,
  } as PlayerState;
}

function feas(id: string, d: number): FeasibilityOut {
  return {
    player_id: id,
    displacement_m: d,
    reachable_radius_m: 10,
    reachable: true,
    origin_x: 0,
    origin_y: 0,
    inside_pitch: true,
    min_separation_m: 2,
    supported: true,
    notes: [],
    kind: "user",
  };
}

/** SYNTHETIC scenario for the "press" situation (made-up numbers). */
function syntheticScenario(over: Partial<ScenarioResponse> = {}): ScenarioResponse {
  return {
    observed: syntheticAnalysis({
      period: 1,
      time_s: 125.0,
      attacking_team: "away",
      players: [
        state("away_17", "away", 17, 10),
        state("away_18", "away", 18, 12),
        state("home_9", "home", 9, 14),
        state("home_4", "home", 4, 20, "estimated"),
        state("home_5", "home", 5, null),
      ],
    }),
    hypothetical: syntheticAnalysis(),
    feasibility: [feas("home_9", 1.5)],
    movements: [],
    option_deltas: [
      {
        receiver_id: "away_18",
        observed_p: 0.11,
        hypothetical_p: 0.22,
        delta: 0.11,
        observed_p10: 0.1,
        observed_p90: 0.12,
        hypothetical_p10: 0.2,
        hypothetical_p90: 0.24,
        delta_sign_agreement: 0.75,
      },
    ],
    summary_observed: {},
    summary_hypothetical: {},
    ensemble_size: 8,
    supported: true,
    warnings: [],
    assumptions: [],
    disclaimer: "Synthetic disclaimer.",
    value_model: "option_model",
    ...over,
  };
}

function defender(id: string, v: number): DefenderShadowOut {
  return {
    player_id: id,
    x: 0,
    y: 0,
    shadow_best: 0,
    shadow_sum: v,
    alt_radius_m: 3.5,
    alternatives: [],
    most_suppressed_receiver: null,
    most_suppressed_delta: 0,
    distinguishable_from_noise: true,
    provenance: "estimated",
  };
}

function syntheticShadow(): DefensiveShadowOut {
  return {
    match_id: PRESS.matchId,
    frame: PRESS.frame,
    attacking_team: "away",
    carrier_id: "away_17",
    value_model: "option_model",
    v_best_observed: 0.5,
    v_sum_observed: 1,
    defenders: [defender("home_4", 0.07), defender("home_9", 0.33)],
    assumptions: [],
    noise_floor: 0.02,
    horizon_s: 1,
    alt_radius_m: 3.5,
    status: "synthetic",
  };
}

const RESPONSE = featuredById("response")!;

function delta(receiver: string, d: number, agreement: number | null) {
  return {
    receiver_id: receiver,
    observed_p: 0.5,
    hypothetical_p: 0.5 + d,
    delta: d,
    observed_p10: 0.45,
    observed_p90: 0.55,
    hypothetical_p10: 0.45 + d,
    hypothetical_p90: 0.55 + d,
    delta_sign_agreement: agreement,
  };
}

/**
 * SYNTHETIC pair for the "response" situation (made-up numbers): the attacker's move with static
 * defenders, and the same move with a marker that follows.
 */
function syntheticResponsePair({ responseDelta = -0.1 }: { responseDelta?: number } = {}): {
  scenario: ScenarioResponse;
  response: ScenarioResponse;
} {
  const observed = syntheticAnalysis({
    period: 1,
    time_s: 125.0,
    attacking_team: "home",
    players: [state("home_1", "home", 1, -40), state("home_5", "home", 5, -29), state("away_22", "away", 22, -28)],
  });
  const userMove = {
    player_id: "home_5",
    from_x: -29,
    from_y: 0,
    to_x: -29,
    to_y: 2.5,
    provenance: "hypothetical" as const,
    reason: "user",
  };
  const base = syntheticScenario({ observed, feasibility: [feas("home_5", 2.5)], movements: [userMove] });
  return {
    scenario: { ...base, option_deltas: [delta("home_5", 0.2, 1)] },
    response: {
      ...base,
      feasibility: [feas("home_5", 2.5), { ...feas("away_22", 1.25), kind: "simulated" }],
      movements: [
        userMove,
        { player_id: "away_22", from_x: -28, from_y: 0, to_x: -28, to_y: 1.25, provenance: "simulated", reason: "synthetic rule" },
      ],
      option_deltas: [delta("home_5", responseDelta, 0.875)],
      policy: {
        name: "marker_follows",
        description: "Synthetic description.",
        assumptions: [],
        validated: false,
        validation_ref: null,
      },
    },
  };
}

const MATCH: MatchSummary = {
  match_id: PRESS.matchId,
  source: SYNTHETIC_SOURCE,
  fps: 25,
  n_frames: 1000,
  periods: [{ period: 1, first_frame: 1, last_frame: 900, duration_s: 36, home_attacks_positive_x: true }],
  n_possessions: 1,
  n_events: 1,
  pitch: PITCH,
  home_label: "Home",
  away_label: "Away",
};

describe("pitchZone (zone grid convention)", () => {
  it("names thirds, penalty areas and channels from the team's attacking direction", () => {
    expect(describeZone(pitchZone(40, 0, 1, PITCH)!)).toBe("Opponent's penalty area, centre");
    expect(describeZone(pitchZone(40, 0, -1, PITCH)!)).toBe("Own penalty area, centre");
    expect(describeZone(pitchZone(0, 0, 1, PITCH)!)).toBe("Middle third, centre");
    expect(describeZone(pitchZone(-20, 0, 1, PITCH)!)).toBe("Defensive third, centre");
    expect(describeZone(pitchZone(25, 25, 1, PITCH)!)).toBe("Attacking third, left wing");
  });

  it("mirrors left and right when the team attacks towards -x", () => {
    expect(pitchZone(0, 15, 1, PITCH)!.channel).toBe("left half-space");
    expect(pitchZone(0, 15, -1, PITCH)!.channel).toBe("right half-space");
    expect(pitchZone(0, -25, -1, PITCH)!.channel).toBe("left wing");
  });

  it("measures the distance to the goal being attacked", () => {
    expect(pitchZone(42.5, 0, 1, PITCH)!.toGoal).toBeCloseTo(10);
    expect(pitchZone(-42.5, 0, -1, PITCH)!.toGoal).toBeCloseTo(10);
  });

  it("returns null without a position or a known direction (never guesses)", () => {
    expect(pitchZone(null, 0, 1, PITCH)).toBeNull();
    expect(pitchZone(0, Number.NaN, 1, PITCH)).toBeNull();
    expect(pitchZone(0, 0, 0, PITCH)).toBeNull();
    expect(pitchZone(0, 0, undefined, PITCH)).toBeNull();
  });
});

describe("featured situations", () => {
  it("carry curated identity and text only, no numbers in the story", () => {
    for (const f of FEATURED) {
      const t = { home: "Home", away: "Away" };
      const text = [f.story(t), ...Object.values(f.steps(t))].join(" ");
      // jersey numbers are identities; no decimals or percentages (results come from the API)
      expect(text).not.toMatch(/\d\.\d|%/);
      expect(featuredById(f.id)).toBe(f);
    }
    expect(featuredById("nope")).toBeNull();
  });

  it("never calls broadcast tracking fully observed", () => {
    const t = { home: "Home", away: "Away" };
    const broadcast = featuredById("broadcast")!;
    const text = [broadcast.story(t), ...Object.values(broadcast.steps(t))].join(" ");
    expect(text).not.toMatch(/(every|all)[^.]*observed/i);
    expect(text).toMatch(/estimate/);
  });

  it("attributes the 'most passing value' claim to the experimental Defensive Shadow model", () => {
    const t = { home: "Home", away: "Away" };
    for (const f of [featuredById("press")!, featuredById("broadcast")!]) {
      expect(f.showShadow).toBe(true);
      expect(f.story(t)).toMatch(/By the experimental Defensive Shadow model, [^.]*most passing value/);
    }
  });

  it("tells the response story as an assumption-dependent model illustration, not as what happened", () => {
    const t = { home: "Home", away: "Away" };
    expect(RESPONSE.response).toBe("marker_follows");
    expect(RESPONSE.showShadow).toBe(false);
    expect(RESPONSE.moved).toBe(RESPONSE.receiver); // the attacker who makes the run
    const story = RESPONSE.story(t);
    expect(story).toMatch(/model-based illustration, not what happened in the match/);
    expect(story).toMatch(/assumption/);
    expect(story).toMatch(/stand still/);
    expect(story).toMatch(/marker follows/);
    expect(story).not.toMatch(/Defensive Shadow|caus|would have/i);
    const steps = RESPONSE.steps(t);
    expect(steps.move).toMatch(/simulated/);
    expect(steps.move).toMatch(/model estimates/);
  });

  it("builds tour and scenario links that round-trip through the router", () => {
    expect(parseRoute(featuredHref(PRESS, "tour"))).toEqual({
      name: "at",
      matchId: PRESS.matchId,
      frame: PRESS.frame,
      tour: "press",
    });
    const sc = parseRoute(featuredHref(PRESS, "scenario"));
    expect(sc).toMatchObject({ name: "at", moves: PRESS.moves });
    expect(sc).not.toHaveProperty("tour");
    expect(sc).not.toHaveProperty("response");
    // a malformed tour id is ignored, never guessed
    expect(parseRoute("#/lab/m%3Ap1?tour=../x")).toEqual({ name: "lab", possessionId: "m:p1" });
    expect(routeHash({ name: "lab", possessionId: "m:p1", frame: 3, tour: "press" })).toBe("#/lab/m%3Ap1?f=3&tour=press");
  });

  it("links the response what-if with its defender response preselected", () => {
    expect(featuredHref(RESPONSE, "scenario")).toBe(
      "#/at/metrica-sample-2?f=31360&move=home_5,-29.04,9.47&response=marker_follows",
    );
    expect(parseRoute(featuredHref(RESPONSE, "scenario"))).toEqual({
      name: "at",
      matchId: RESPONSE.matchId,
      frame: RESPONSE.frame,
      moves: RESPONSE.moves,
      response: "marker_follows",
    });
    // the tour applies the response in step 3 itself
    expect(parseRoute(featuredHref(RESPONSE, "tour"))).not.toHaveProperty("response");
    // malformed policy names are ignored, never guessed
    expect(parseRoute("#/lab/m%3Ap1?response=Marker%20follows")).toEqual({ name: "lab", possessionId: "m:p1" });
    expect(parseRoute("#/lab/m%3Ap1?response=../x")).toEqual({ name: "lab", possessionId: "m:p1" });
  });

  it("reads every fact from the API payloads (synthetic)", () => {
    const f = featuredFacts(PRESS, { scenario: syntheticScenario(), shadow: syntheticShadow(), match: MATCH });
    expect(f.clock).toBe("P1 02:04"); // (125.0 s - 1/25 s period start) on the match clock
    expect(f.pass?.observed_p).toBe(0.11);
    expect(f.agreeing).toBe(6); // 0.75 of 8
    expect(f.displacement_m).toBe(1.5);
    expect(f.shadow).toEqual({ value: 0.33, next: 0.07, isTop: true });
    expect(f.response).toBeNull(); // this situation has no defender response
    expect(f.shown).toBe(4); // home_5 has no position
    expect(f.estimated).toBe(1);
  });

  it("leaves unknown facts empty instead of inventing them", () => {
    const f = featuredFacts(PRESS, { scenario: syntheticScenario({ option_deltas: [], feasibility: [] }) });
    expect(f.clock).toBeNull();
    expect(f.pass).toBeNull();
    expect(f.agreeing).toBeNull();
    expect(f.displacement_m).toBeNull();
    expect(f.shadow).toBeNull();
    expect(f.response).toBeNull();
    // a shadow that does not include the moved defender gives no shadow fact
    const other = { ...syntheticShadow(), defenders: [defender("home_4", 0.07)] };
    expect(featuredFacts(PRESS, { scenario: syntheticScenario(), shadow: other }).shadow).toBeNull();
    // a single defender has nothing to be compared with
    const alone = { ...syntheticShadow(), defenders: [defender("home_9", 0.33)] };
    expect(featuredFacts(PRESS, { scenario: syntheticScenario(), shadow: alone }).shadow).toEqual({
      value: 0.33,
      next: null,
      isTop: null,
    });
  });

  it("compares static defenders with the defender response for a moved attacker (synthetic)", () => {
    const { scenario, response } = syntheticResponsePair();
    const f = featuredFacts(RESPONSE, { scenario, response, shadow: syntheticShadow(), match: MATCH });
    expect(f.shadow).toBeNull(); // the moved player is an attacker: no shadow fact, even if one is supplied
    expect(f.displacement_m).toBe(2.5);
    expect(f.pass?.delta).toBe(0.2);
    expect(f.agreeing).toBe(8);
    expect(f.response).toMatchObject({
      policy: "marker_follows",
      agreeing: 7,
      ensembleSize: 8,
      supported: true,
      flips: true,
      simulated: [{ player_id: "away_22", distance_m: 1.25 }],
    });
    expect(f.response?.pass?.delta).toBe(-0.1);
    expect(f.response?.info?.validated).toBe(false);
    // the user's move is hypothetical, not a simulated response
    expect(f.response?.simulated.map((m) => m.player_id)).not.toContain("home_5");
  });

  it("does not call a direction flipped when a change is negligible or missing", () => {
    const { scenario, response } = syntheticResponsePair({ responseDelta: 0.001 });
    expect(featuredFacts(RESPONSE, { scenario, response }).response?.flips).toBeNull();
    const same = syntheticResponsePair({ responseDelta: 0.05 });
    expect(featuredFacts(RESPONSE, same).response?.flips).toBe(false);
    const missing = syntheticResponsePair();
    const r = featuredFacts(RESPONSE, { ...missing, response: { ...missing.response, option_deltas: [] } }).response;
    expect(r?.pass).toBeNull();
    expect(r?.flips).toBeNull();
    // without the response payload there is no response fact at all
    expect(featuredFacts(RESPONSE, { scenario: missing.scenario }).response).toBeNull();
  });
});

describe("FeaturedCard (mocked API, synthetic payloads)", () => {
  beforeEach(() => resetFeaturedLive());
  afterEach(() => vi.restoreAllMocks());

  it("shows a loading state, then the live numbers with provenance labels", async () => {
    vi.spyOn(api, "scenario").mockResolvedValue(syntheticScenario());
    vi.spyOn(api, "defensiveShadow").mockResolvedValue(syntheticShadow());
    render(<FeaturedCard f={PRESS} match={MATCH} testId="featured-link" />);
    expect(screen.getByTestId("featured-loading")).toBeInTheDocument();
    const facts = await screen.findByTestId("featured-facts");
    expect(await within(facts).findByText("0.33")).toBeInTheDocument();
    expect(facts).toHaveTextContent("0.11 →to 0.22");
    expect(facts).toHaveTextContent("+0.11");
    expect(facts).toHaveTextContent("6 of 8 model settings agree");
    expect(facts).toHaveTextContent("1 estimated (off camera)");
    expect(within(facts).getAllByText("estimated").length).toBeGreaterThan(0);
    expect(within(facts).getByText("hypothetical")).toBeInTheDocument();
    expect(screen.getByText(/P1 02:04/)).toBeInTheDocument();
    expect(screen.getByTestId("featured-link")).toHaveAttribute("href", featuredHref(PRESS, "scenario"));
    expect(screen.getByTestId("tour-link-press")).toHaveAttribute("href", featuredHref(PRESS, "tour"));
  });

  it("keeps the card usable when the shadow is unavailable", async () => {
    vi.spyOn(api, "scenario").mockResolvedValue(syntheticScenario());
    vi.spyOn(api, "defensiveShadow").mockRejectedValue(new Error("down"));
    render(<FeaturedCard f={PRESS} match={MATCH} testId="featured-link" />);
    const facts = await screen.findByTestId("featured-facts");
    await vi.waitFor(() => expect(screen.queryByTestId("featured-shadow-loading")).toBeNull());
    expect(facts).not.toHaveTextContent("Defensive Shadow");
    expect(facts).toHaveTextContent("0.11");
  });

  it("explains a failed computation and retries", async () => {
    const sc = vi.spyOn(api, "scenario").mockRejectedValueOnce(new Error("boom")).mockResolvedValue(syntheticScenario());
    vi.spyOn(api, "defensiveShadow").mockResolvedValue(syntheticShadow());
    render(<FeaturedCard f={PRESS} match={MATCH} testId="featured-link" />);
    expect(await screen.findByText("Live numbers unavailable")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("featured-facts")).toBeInTheDocument();
    expect(sc).toHaveBeenCalledTimes(2);
  });

  it("shows static defenders against the defender response, with the simulated move labelled", async () => {
    const pair = syntheticResponsePair();
    const sc = vi
      .spyOn(api, "scenario")
      .mockImplementation((req) => Promise.resolve(req.response_policy === "none" ? pair.scenario : pair.response));
    const sh = vi.spyOn(api, "defensiveShadow").mockResolvedValue(syntheticShadow());
    render(<FeaturedCard f={RESPONSE} match={MATCH} testId="featured-response-link" />);
    const cmp = await screen.findByTestId("featured-response");
    // the same move, requested with static defenders and with the situation's response
    expect(sc.mock.calls.map(([req]) => req.response_policy).sort()).toEqual(["marker_follows", "none"]);
    for (const [req] of sc.mock.calls) expect(req.interventions).toEqual(RESPONSE.moves);
    expect(sh).not.toHaveBeenCalled();
    expect(screen.queryByText(/Defensive Shadow/)).toBeNull();
    expect(cmp).toHaveTextContent("#5 Home moved 2.5 m");
    expect(within(cmp).getByTestId("featured-response-static")).toHaveTextContent("Defenders stand still0.50 →to 0.70+0.20");
    const resp = within(cmp).getByTestId("featured-response-response");
    expect(resp).toHaveTextContent("Marker follows");
    expect(resp).toHaveTextContent("0.50 →to 0.40−0.10");
    expect(within(resp).getByText("experimental")).toBeInTheDocument();
    expect(within(cmp).getByTestId("featured-response-verdict")).toHaveTextContent(
      "The direction of the change depends on the assumption about the defence.",
    );
    const sim = within(cmp).getByTestId("featured-simulated-move");
    expect(sim).toHaveTextContent("#22 Away moved 1.3 m by the Marker follows response");
    expect(within(sim).getByText("simulated")).toHaveClass("prov--simulated");
    expect(within(cmp).getByText("hypothetical")).toBeInTheDocument();
    expect(cmp).toHaveTextContent("model settings agreeing on the direction: 8 of 8 static, 7 of 8 Marker follows");
    expect(screen.getByTestId("featured-response-link")).toHaveAttribute("href", featuredHref(RESPONSE, "scenario"));
  });

  it("shows the response card's failure when either scenario fails, and retries", async () => {
    const pair = syntheticResponsePair();
    let failed = false;
    const sc = vi.spyOn(api, "scenario").mockImplementation((req) => {
      if (req.response_policy !== "none" && !failed) {
        failed = true;
        return Promise.reject(new Error("boom"));
      }
      return Promise.resolve(req.response_policy === "none" ? pair.scenario : pair.response);
    });
    render(<FeaturedCard f={RESPONSE} match={MATCH} testId="x" />);
    expect(await screen.findByText("Live numbers unavailable")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("featured-response")).toBeInTheDocument();
    // the static result was kept; only the failed response is asked again
    expect(sc).toHaveBeenCalledTimes(3);
  });

  it("asks the server once per situation, even when mounted twice", async () => {
    const sc = vi.spyOn(api, "scenario").mockResolvedValue(syntheticScenario());
    vi.spyOn(api, "defensiveShadow").mockResolvedValue(syntheticShadow());
    const { unmount } = render(<FeaturedCard f={PRESS} match={MATCH} testId="a" />);
    await screen.findByTestId("featured-facts");
    unmount();
    render(<FeaturedCard f={PRESS} match={MATCH} testId="b" />);
    await screen.findByTestId("featured-facts");
    expect(sc).toHaveBeenCalledTimes(1);
  });
});

describe("GuideBar", () => {
  const teams = { home: "Home", away: "Away" };
  const base = {
    featured: PRESS,
    teams,
    playing: false,
    player: "#9 Home",
    response: null,
    moved: false,
    onStep: vi.fn(),
    onWatch: vi.fn(),
    onJump: vi.fn(),
    onExit: vi.fn(),
  };

  it("marks the current step and offers its action", async () => {
    const onWatch = vi.fn();
    const onStep = vi.fn();
    render(<GuideBar {...base} step={1} onWatch={onWatch} onStep={onStep} />);
    expect(screen.getByTestId("guide-step-1")).toHaveAttribute("aria-current", "step");
    expect(screen.getByTestId("guide-step-2")).not.toHaveAttribute("aria-current");
    await userEvent.click(screen.getByTestId("guide-watch"));
    expect(onWatch).toHaveBeenCalled();
    await userEvent.click(screen.getByTestId("guide-next"));
    expect(onStep).toHaveBeenCalledWith(2);
  });

  it("points to the evidence and similar moments in step 2", async () => {
    const onJump = vi.fn();
    render(<GuideBar {...base} step={2} onJump={onJump} />);
    await userEvent.click(screen.getByRole("button", { name: "Similar moments" }));
    expect(onJump).toHaveBeenCalledWith("similar");
    expect(screen.getByTestId("guide-next")).toHaveTextContent("Next: move #9 Home");
  });

  it("explains the views and the response selector once the move is applied", async () => {
    const onExit = vi.fn();
    const { rerender } = render(<GuideBar {...base} step={3} onExit={onExit} />);
    expect(screen.queryByText(/drag an attacker/)).toBeNull();
    rerender(<GuideBar {...base} step={3} moved onExit={onExit} />);
    expect(screen.getByText(/drag an\s+attacker/)).toBeInTheDocument();
    await userEvent.click(screen.getByTestId("guide-finish"));
    expect(onExit).toHaveBeenCalled();
    // a static-only situation offers no static-vs-response comparison
    expect(screen.queryByTestId("guide-compare")).toBeNull();
    expect(screen.queryByTestId("guide-response")).toBeNull();
  });

  it("points to the static-vs-response comparison when the situation has a defender response", async () => {
    const onJump = vi.fn();
    const props = { ...base, featured: RESPONSE, player: "#5 Home", response: "Marker follows", onJump };
    const { rerender } = render(<GuideBar {...props} step={3} />);
    expect(screen.queryByTestId("guide-response")).toBeNull();
    rerender(<GuideBar {...props} step={3} moved />);
    const text = screen.getByTestId("guide-response");
    expect(text).toHaveTextContent("set to Marker follows");
    expect(text).toHaveTextContent("simulated, not observed");
    expect(text).toHaveTextContent("Static (no response)");
    expect(screen.queryByText(/drag an\s+attacker/)).toBeNull();
    await userEvent.click(screen.getByTestId("guide-compare"));
    expect(onJump).toHaveBeenCalledWith("compare");
  });
});

describe("PlayerInspector (F11)", () => {
  const player: PlayerOut = { player_id: "away_17", team: "away", jersey: 17 };
  const info = {
    player,
    x: 40,
    y: 15,
    zone: pitchZone(40, 15, 1, PITCH),
    speed: 3.2,
    speedSource: "(synthetic)",
    positionProv: "estimated" as const,
    velocityProv: "derived" as const,
    isGk: false,
  };

  it("describes where the player is in football terms, not by raw id", () => {
    render(<PlayerInspector info={info} playing={false} editable={false} onMove={vi.fn()} onClose={vi.fn()} />);
    const where = screen.getByTestId("player-zone");
    expect(where).toHaveTextContent("Opponent's penalty area, left half-space");
    expect(within(where).getByText("estimated")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /#17 Away/ })).toBeInTheDocument();
    expect(screen.queryByText("away_17")).toBeNull();
    // coordinates only in the analyst disclosure
    expect(screen.getByText("Pitch coordinates")).toBeInTheDocument();
    expect(screen.getByTestId("player-coords")).toHaveTextContent("x 40.0 m, y 15.0 m");
  });

  it("says when the direction of play is unknown instead of guessing a zone", () => {
    render(
      <PlayerInspector info={{ ...info, zone: null }} playing={false} editable={false} onMove={vi.fn()} onClose={vi.fn()} />,
    );
    expect(screen.getByTestId("player-zone")).toHaveTextContent("direction of play unknown");
    expect(screen.queryByText("To goal")).toBeNull();
  });
});

describe("ProvenanceKey", () => {
  it("explains all five labels with their badges", () => {
    render(<ProvenanceKey />);
    const key = screen.getByTestId("provenance-key");
    for (const p of ["observed", "derived", "estimated", "hypothetical", "simulated"]) {
      expect(within(key).getByText(p)).toHaveClass(`prov--${p}`);
    }
  });
});

describe("ScenarioPanel: response hint for a defender-only edit", () => {
  const players = new Map<string, PlayerOut>([
    ["home_9", { player_id: "home_9", team: "home", jersey: 9 }],
    ["away_18", { player_id: "away_18", team: "away", jersey: 18 }],
  ]);
  const props = (interventions: { player_id: string; x: number; y: number }[]) => ({
    ready: true,
    frameLabel: "P1 00:00",
    interventions,
    policy: "marker_follows",
    onPolicy: vi.fn(),
    policies: [],
    staticStatus: "success" as const,
    staticScenario: null,
    valueModel: "option_model" as const,
    onValueModel: vi.fn(),
    status: "success" as const,
    scenario: syntheticScenario(), // attacking team: away
    error: null,
    onRetry: vi.fn(),
    onRemove: vi.fn(),
    onReset: vi.fn(),
    players,
    highlight: null,
    onHighlight: vi.fn(),
    notice: null,
  });

  it("explains why no defender responded when only defenders were moved", () => {
    render(<ScenarioPanel {...props([{ player_id: "home_9", x: 1, y: 1 }])} />);
    expect(screen.getByTestId("no-simulated-moves")).toBeInTheDocument();
    expect(screen.getByTestId("response-hint")).toHaveTextContent("Move an attacker");
  });

  it("does not show the hint when an attacker was moved", () => {
    render(<ScenarioPanel {...props([{ player_id: "away_18", x: 1, y: 1 }])} />);
    expect(screen.getByTestId("no-simulated-moves")).toBeInTheDocument();
    expect(screen.queryByTestId("response-hint")).toBeNull();
  });
});
