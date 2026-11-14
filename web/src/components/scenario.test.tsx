/**
 * Scenario panel: defender response selector, simulated moves and the static-vs-policy
 * comparison. All payloads are SYNTHETIC.
 */
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { OptionDeltaOut, PlayerOut, PolicyInfo, ScenarioResponse } from "../api/types";
import { syntheticAnalysis } from "../test/fixtures";
import { FALLBACK_POLICIES } from "../lib/policies";
import { PolicyComparison } from "./PolicyComparison";
import { ScenarioPanel } from "./ScenarioPanel";
import { StatusBlock } from "./StatusBlock";

const players = new Map<string, PlayerOut>([
  ["home_7", { player_id: "home_7", team: "home", jersey: 7 }],
  ["home_9", { player_id: "home_9", team: "home", jersey: 9 }],
  ["away_4", { player_id: "away_4", team: "away", jersey: 4 }],
]);

function delta(id: string, d: number): OptionDeltaOut {
  return {
    receiver_id: id,
    observed_p: 0.5,
    hypothetical_p: 0.5 + d,
    delta: d,
    observed_p10: 0.4,
    observed_p90: 0.6,
    hypothetical_p10: 0.4 + d,
    hypothetical_p90: 0.6 + d,
    delta_sign_agreement: 1,
  };
}

const PRESS: PolicyInfo = {
  name: "press_carrier",
  description: "Synthetic: the nearest defender steps towards the carrier.",
  assumptions: ["synthetic press assumption"],
  validated: false,
  validation_ref: null,
};

function scenario(over: Partial<ScenarioResponse> = {}): ScenarioResponse {
  return {
    observed: syntheticAnalysis(),
    hypothetical: syntheticAnalysis(),
    feasibility: [],
    movements: [{ player_id: "home_7", from_x: 0, from_y: 0, to_x: 2, to_y: 0, provenance: "hypothetical", reason: "user" }],
    option_deltas: [delta("home_9", 0.1)],
    summary_observed: {},
    summary_hypothetical: {},
    ensemble_size: 24,
    supported: true,
    warnings: [],
    assumptions: [],
    disclaimer: "Synthetic disclaimer: not what would have happened in the match.",
    value_model: "option_model",
    ...over,
  };
}

function panel(over: Partial<Parameters<typeof ScenarioPanel>[0]> = {}) {
  const props: Parameters<typeof ScenarioPanel>[0] = {
    ready: true,
    frameLabel: "P1 00:04",
    interventions: [{ player_id: "home_7", x: 2, y: 0 }],
    policy: "none",
    onPolicy: () => {},
    policies: FALLBACK_POLICIES,
    staticStatus: "idle",
    staticScenario: null,
    valueModel: "option_model",
    onValueModel: () => {},
    status: "success",
    scenario: scenario(),
    error: null,
    onRetry: () => {},
    onRemove: () => {},
    onReset: () => {},
    players,
    highlight: null,
    onHighlight: () => {},
    notice: null,
    ...over,
  };
  return render(<ScenarioPanel {...props} />);
}

describe("ScenarioPanel: defender response", () => {
  it("offers the fallback policies on an older backend, static first, with no experimental badge for static", () => {
    panel();
    const select = screen.getByLabelText("Defender response");
    const options = within(select).getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["Static (no response)", "Marker follows (experimental)"]);
    expect(screen.queryByTestId("policy-badge")).toBeNull();
    expect(screen.getByTestId("policy-static-hint")).toHaveTextContent("Choose a defender response");
  });

  it("lists backend policies and reports the choice", async () => {
    const onPolicy = vi.fn();
    panel({ policies: [...FALLBACK_POLICIES, PRESS], onPolicy });
    await userEvent.selectOptions(screen.getByTestId("policy-select"), "press_carrier");
    expect(onPolicy).toHaveBeenCalledWith("press_carrier");
  });

  it("badges an unvalidated policy as experimental and labels simulated moves", () => {
    const sc = scenario({
      policy: PRESS,
      movements: [
        { player_id: "home_7", from_x: 0, from_y: 0, to_x: 2, to_y: 0, provenance: "hypothetical", reason: "user" },
        { player_id: "away_4", from_x: 5, from_y: 0, to_x: 8, to_y: 4, provenance: "simulated", reason: "steps to the carrier" },
      ],
    });
    panel({ policy: "press_carrier", policies: [...FALLBACK_POLICIES, PRESS], scenario: sc, staticStatus: "loading" });
    // no status from the backend: the safe fallback badge
    expect(screen.getByTestId("policy-badge")).toHaveTextContent("experimental");
    expect(screen.getByTestId("policy-description")).toHaveTextContent("No verdict from an observational check");
    const sim = screen.getByTestId("simulated-move");
    expect(sim).toHaveTextContent("simulated");
    expect(sim).toHaveTextContent("#4 Away moved 5.0 m by the Press the carrier response");
    expect(sim).toHaveTextContent("steps to the carrier");
    expect(screen.getByText("Computing the same edit with static defenders")).toBeInTheDocument();
    expect(screen.getByTestId("policy-assumptions")).toHaveTextContent("synthetic press assumption");
  });

  it("shows a validated policy's verdict, badge and evidence from the backend's status", () => {
    const validated: PolicyInfo = {
      ...PRESS,
      validated: true,
      validation_ref: "CLAIMS_LEDGER C99 (synthetic)",
      status: "validated: synthetic forecasts beat a synthetic baseline",
    };
    panel({ policy: "press_carrier", policies: [validated], scenario: scenario({ policy: validated }) });
    expect(screen.getByTestId("policy-badge")).toHaveTextContent("forecaster");
    expect(screen.getByTestId("policy-status")).toHaveTextContent(
      "Backend verdict: validated: synthetic forecasts beat a synthetic baseline.",
    );
    expect(screen.getByTestId("policy-description")).toHaveTextContent("Evidence: CLAIMS_LEDGER C99 (synthetic)");
  });

  it("labels every selector option and the chosen policy with the badge its status implies", () => {
    // SYNTHETIC descriptions; the status strings follow the backend's vocabulary
    const mk = (name: string, validated: boolean, status: string): PolicyInfo => ({
      name,
      description: `Synthetic ${name}.`,
      assumptions: [],
      validated,
      validation_ref: null,
      status,
    });
    const list = [
      mk("none", false, "no response simulated"),
      mk("marker_follows", false, "experimental: consistent with real defender reactions in an observational check"),
      mk("press_carrier", false, "experimental: not supported by the observational check"),
      mk("zonal_shift", false, "experimental: overshoots real defensive movement in the observational check"),
      mk("learned", true, "validated: short-horizon forecasts of real play beat constant velocity"),
      mk("learned_innovation", true, "validated (observational): forecasts beat constant velocity and a placebo"),
      mk("future_rule", false, "experimental: a new rule"),
    ];
    const { unmount } = panel({ policies: list });
    const select = screen.getByTestId("policy-select");
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Static (no response)",
      "Marker follows (observational)",
      "Press the carrier (not supported)",
      "Zonal shift (not supported)",
      "Learned forecaster (forecaster)",
      "Learned reaction to attackers (observational)",
      "Future rule (experimental)",
    ]);
    // the static baseline has no badge and no verdict note
    expect(screen.queryByTestId("policy-badge")).toBeNull();
    expect(screen.queryByTestId("policy-verdict-note")).toBeNull();
    unmount();

    panel({ policy: "zonal_shift", policies: list, scenario: scenario({ policy: list[3] }) });
    const badge = screen.getByTestId("policy-badge");
    expect(badge).toHaveTextContent("not supported");
    expect(badge).toHaveAttribute("title", expect.stringMatching(/does not support this rule/));
    expect(screen.getByTestId("policy-status")).toHaveTextContent("overshoots real defensive movement");
    // never "not validated against observed defender movement" for a rule the check addressed
    expect(screen.getByTestId("policy-description")).not.toHaveTextContent(/Not validated against observed/);
  });

  it("calls an observational verdict non-causal", () => {
    const marker: PolicyInfo = {
      ...PRESS,
      name: "marker_follows",
      status: "experimental: consistent with real defender reactions in an observational check (synthetic)",
    };
    panel({ policy: "marker_follows", policies: [marker], scenario: scenario({ policy: marker }) });
    expect(screen.getByTestId("policy-badge")).toHaveTextContent("observational");
    expect(screen.getByTestId("policy-verdict-note")).toHaveTextContent("not a causal effect");
  });

  it("warns that learned_innovation extrapolates beyond its fitted range, citing the edit that exceeds it", () => {
    // SYNTHETIC policy; the fitted range is read from its assumptions text, as the API words it
    const innovation: PolicyInfo = {
      name: "learned_innovation",
      description: "Synthetic learned reaction.",
      assumptions: [
        "Linear: a relocation larger than real unexpected movement (99% of attacker innovations are below about 2.5 m in 1 s) is an extrapolation.",
      ],
      validated: true,
      validation_ref: null,
      status: "validated (observational): synthetic",
    };
    const feas = (id: string, d: number) => ({
      player_id: id,
      displacement_m: d,
      reachable_radius_m: 10.5,
      reachable: true,
      origin_x: 0,
      origin_y: 0,
      inside_pitch: true,
      min_separation_m: 2,
      supported: true,
      notes: [],
      kind: "user" as const,
      kinematic_reachable: false,
    });
    const sc = scenario({ policy: innovation, feasibility: [feas("home_7", 4.2), feas("home_9", 2.54)] });
    panel({ policy: "learned_innovation", policies: [innovation], scenario: sc });
    expect(screen.getByTestId("policy-badge")).toHaveTextContent("observational");
    const fit = screen.getByTestId("policy-fit-range");
    expect(fit).toHaveTextContent("99% of the real unexpected attacker movement it was fitted on is under about 2.5 m in 1 s");
    expect(fit).toHaveTextContent("Edits larger than about 2.5 m");
    expect(screen.getByTestId("policy-fit-exceeded")).toHaveTextContent("This edit moves #7 Home 4.2 m.");
    expect(screen.getByTestId("policy-fit-exceeded")).not.toHaveTextContent("#9");
    // the reach radius and the horizon come from the response and the request, not app constants
    const detail = screen.getAllByTestId("feasibility-detail")[0]!;
    expect(detail).toHaveTextContent("within 10.5 m of the observed position 1.5 s earlier");
    expect(detail).toHaveTextContent("outside the kinematic envelope (diagnostic only)");
  });

  it("shows no fitted-range warning for a policy that states none", () => {
    const marker: PolicyInfo = { ...PRESS, name: "marker_follows", status: "experimental: consistent with real defender reactions in an observational check" };
    panel({ policy: "marker_follows", policies: [marker], scenario: scenario({ policy: marker }) });
    expect(screen.queryByTestId("policy-fit-range")).toBeNull();
  });

  it("keeps a chosen policy selectable even if the list does not contain it", () => {
    panel({ policy: "lane_cover", policies: FALLBACK_POLICIES, scenario: scenario({ movements: [] }) });
    expect(within(screen.getByTestId("policy-select")).getByRole("option", { name: /Cover passing lanes/ })).toBeInTheDocument();
    expect(screen.getByTestId("no-simulated-moves")).toHaveTextContent("moved no defender");
  });

  it("compares static and policy deltas once both are available", () => {
    const sc = scenario({
      policy: PRESS,
      option_deltas: [delta("home_9", -0.06), delta("home_7", 0.02)],
      movements: [{ player_id: "away_4", from_x: 5, from_y: 0, to_x: 8, to_y: 4, provenance: "simulated", reason: "r" }],
    });
    const st = scenario({ option_deltas: [delta("home_9", 0.1), delta("home_7", 0.03)] });
    panel({ policy: "press_carrier", policies: [PRESS], scenario: sc, staticStatus: "success", staticScenario: st });
    const cmp = screen.getByTestId("policy-comparison");
    expect(cmp).toHaveAttribute("data-verdict", "changes");
    expect(screen.getByTestId("policy-verdict")).toHaveTextContent("1 of 2 pass options change direction");
    expect(within(cmp).getByText("direction flips")).toBeInTheDocument();
  });
});

describe("PolicyComparison", () => {
  it("says the result is unchanged when the policy moved nobody", () => {
    render(
      <PolicyComparison policy="marker_follows" staticDeltas={[delta("home_9", 0.1)]} policyDeltas={[delta("home_9", 0.1)]} players={players} nSimulated={0} />,
    );
    expect(screen.getByTestId("policy-verdict")).toHaveTextContent("moved no defender");
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows the five largest changes and can expand to all", async () => {
    const ids = ["a", "b", "c", "d", "e", "f", "g"];
    render(
      <PolicyComparison
        policy="zonal_shift"
        staticDeltas={ids.map((id, i) => delta(id, 0.01 * (i + 1)))}
        policyDeltas={ids.map((id, i) => delta(id, 0.01 * (i + 1) + 0.001))}
        players={players}
        nSimulated={2}
      />,
    );
    expect(screen.getAllByRole("row")).toHaveLength(1 + 5);
    await userEvent.click(screen.getByRole("button", { name: "Show all 7 options" }));
    expect(screen.getAllByRole("row")).toHaveLength(1 + 7);
  });
});

describe("StatusBlock with an error", () => {
  it("explains a missing item without a pointless retry, and keeps the technical text", () => {
    render(
      <StatusBlock kind="error" title="Could not load this possession" error={new ApiError(404, "unknown possession 'x'")} subject="this possession" onRetry={() => {}} />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("We could not find this possession");
    expect(screen.getByRole("alert")).toHaveTextContent("HTTP 404: unknown possession 'x'");
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("offers retry when the server is unreachable", () => {
    render(<StatusBlock kind="error" title="Could not load matches" error={new ApiError(0, "down", true)} onRetry={() => {}} />);
    expect(screen.getByRole("alert")).toHaveTextContent("not responding");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("says a hung server timed out and offers retry instead of spinning", async () => {
    const onRetry = vi.fn();
    const err = new ApiError(0, "The FootyMind API did not answer within 30 s (timed out).", true, true);
    render(<StatusBlock kind="error" title="Scenario failed" error={err} onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("did not answer in time (timed out)");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
