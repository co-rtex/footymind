import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { OptionDeltaOut, PassOptionOut, PlayerOut } from "../api/types";
import { Notices } from "./Notices";
import { OptionDeltaTable } from "./OptionDeltaTable";
import { PassOptionsTable } from "./PassOptionsTable";
import { Segmented } from "./Segmented";
import { StatusBlock } from "./StatusBlock";

// SYNTHETIC players and options for component tests.
const players = new Map<string, PlayerOut>([
  ["home_7", { player_id: "home_7", team: "home", jersey: 7 }],
  ["home_9", { player_id: "home_9", team: "home", jersey: 9 }],
  ["away_4", { player_id: "away_4", team: "away", jersey: 4 }],
]);

function option(id: string, p: number, model: number | null = p): PassOptionOut {
  return {
    receiver_id: id,
    target_x: 0,
    target_y: 0,
    distance_m: 12.3,
    forward_progress_m: -4.2,
    ball_travel_time_s: 0.8,
    p_control_at_target: 0.9,
    p_lane_clear: 0.8,
    p_success: p,
    p_success_model: model,
    lane_min_margin_s: 0.3,
    lane_closest_defender_id: "away_4",
    lane_closest_defender_perp_m: 2.5,
    defenders_within_3m_of_lane: 1,
    provenance: "estimated",
  };
}

describe("PassOptionsTable", () => {
  it("ranks options by p_success and names the lane defender", () => {
    render(<PassOptionsTable options={[option("home_9", 0.3), option("home_7", 0.92)]} players={players} highlight={null} onHighlight={() => {}} modelLabel="Option model" />);
    const rows = within(screen.getByTestId("pass-options")).getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("#7 Home");
    expect(rows[0]).toHaveTextContent("0.92");
    expect(rows[0]).toHaveTextContent("#4 Away · 2.5 m");
    expect(rows[1]).toHaveTextContent("#9 Home");
  });

  it("ranks by the calibrated option model, not the physics heuristic", () => {
    render(
      <PassOptionsTable
        options={[option("home_7", 0.95, 0.4), option("home_9", 0.2, 0.8)]}
        players={players}
        highlight={null}
        onHighlight={() => {}}
        modelLabel="Option model"
      />,
    );
    const rows = within(screen.getByTestId("pass-options")).getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("#9 Home");
    expect(rows[0]).toHaveTextContent("0.80");
    expect(rows[0]).toHaveTextContent("0.20");
  });

  it("falls back to physics ranking and says so when the option model is missing", () => {
    render(
      <PassOptionsTable
        options={[option("home_7", 0.3, null), option("home_9", 0.6, null)]}
        players={players}
        highlight={null}
        onHighlight={() => {}}
        modelLabel="Option model"
      />,
    );
    expect(screen.getByRole("columnheader", { name: "Model n/a" })).toBeInTheDocument();
    const rows = within(screen.getByTestId("pass-options")).getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("#9 Home");
  });

  it("handles an empty option list", () => {
    render(<PassOptionsTable options={[]} players={players} highlight={null} onHighlight={() => {}} modelLabel="Option model" />);
    expect(screen.getByText("No pass options at this frame.")).toBeInTheDocument();
  });

  it("highlights a receiver from the keyboard", async () => {
    const onHighlight = vi.fn();
    render(<PassOptionsTable options={[option("home_7", 0.92)]} players={players} highlight={null} onHighlight={onHighlight} modelLabel="Option model" />);
    await userEvent.tab();
    expect(onHighlight).toHaveBeenCalledWith("home_7");
  });
});

describe("OptionDeltaTable", () => {
  const d = (id: string, delta: number | null, agree: number | null): OptionDeltaOut => ({
    receiver_id: id,
    observed_p: 0.5,
    hypothetical_p: delta === null ? null : 0.5 + delta,
    delta,
    observed_p10: 0.4,
    observed_p90: 0.6,
    hypothetical_p10: 0.45,
    hypothetical_p90: 0.7,
    delta_sign_agreement: agree,
  });

  it("orders by |delta|, shows signs, interval bars and agreement", () => {
    render(
      <OptionDeltaTable
        deltas={[d("home_9", 0.0001, 0.46), d("home_7", -0.12, 0.96)]}
        players={players}
        ensembleSize={24}
        highlight={null}
        onHighlight={() => {}}
      />,
    );
    const rows = within(screen.getByTestId("option-deltas")).getAllByRole("row").slice(1);
    expect(rows[0]).toHaveTextContent("#7 Home");
    expect(rows[0]).toHaveTextContent("−0.12");
    expect(rows[0]).toHaveTextContent("96% robust");
    // a negligible delta must not claim a sign agreement
    expect(rows[1]).toHaveTextContent("negligible");
    expect(rows[1]).not.toHaveTextContent("46%");
    expect(screen.getAllByRole("img")[0]).toHaveAccessibleName(/Observed 0\.50/);
  });
});

describe("Notices and Segmented", () => {
  it("shows API warnings verbatim", () => {
    render(<Notices warnings={["ball position not observed at this frame; pass options unavailable"]} />);
    expect(screen.getByText("ball position not observed at this frame; pass options unavailable")).toBeInTheDocument();
  });

  it("renders nothing without warnings", () => {
    const { container } = render(<Notices warnings={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("is a keyboard-operable radio group", async () => {
    const onChange = vi.fn();
    render(
      <Segmented
        label="Speed"
        value="1"
        onChange={onChange}
        options={[
          { value: "0.5", label: "0.5x" },
          { value: "1", label: "1x" },
        ]}
      />,
    );
    const on = screen.getByRole("radio", { name: "1x" });
    expect(on).toHaveAttribute("aria-checked", "true");
    on.focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(onChange).toHaveBeenCalledWith("0.5");
  });
});

describe("StatusBlock", () => {
  it("is the page's level-one heading when it stands for the whole page", () => {
    render(<StatusBlock kind="error" title="Could not open this moment" pageHeading />);
    expect(screen.getByRole("heading", { level: 1, name: "Could not open this moment" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("is not a heading inside a panel", () => {
    render(<StatusBlock kind="loading" title="Loading possessions" />);
    expect(screen.queryByRole("heading")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent("Loading possessions");
  });
});
