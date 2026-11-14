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
    expect(screen.queryByTestId("policy-experimental")).toBeNull();
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
    expect(screen.getByTestId("policy-experimental")).toHaveTextContent("experimental");
    expect(screen.getByTestId("policy-description")).toHaveTextContent("Not validated against observed defender movement");
    const sim = screen.getByTestId("simulated-move");
    expect(sim).toHaveTextContent("simulated");
    expect(sim).toHaveTextContent("#4 Away moved 5.0 m by the Press the carrier response");
    expect(sim).toHaveTextContent("steps to the carrier");
    expect(screen.getByText("Computing the same edit with static defenders")).toBeInTheDocument();
    expect(screen.getByTestId("policy-assumptions")).toHaveTextContent("synthetic press assumption");
  });

  it("shows the validation reference instead of the badge for a validated policy", () => {
    const validated = { ...PRESS, validated: true, validation_ref: "CLAIMS_LEDGER C99 (synthetic)" };
    panel({ policy: "press_carrier", policies: [validated], scenario: scenario({ policy: validated }) });
    expect(screen.queryByTestId("policy-experimental")).toBeNull();
    expect(screen.getByTestId("policy-description")).toHaveTextContent("Validated: CLAIMS_LEDGER C99 (synthetic)");
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
});
