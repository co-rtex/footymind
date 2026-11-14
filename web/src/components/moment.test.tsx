/**
 * "Why this moment" and "Similar moments": parse contract, grouping, empty states and panels.
 * Every payload here is SYNTHETIC (made-up statements, neighbours and outcomes).
 */
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { ContractError, parseAnalysis, parseSimilar } from "../api/parse";
import type { PlayerOut, SimilarMomentsOut, StatementOut } from "../api/types";
import { groupStatements } from "../lib/explain";
import { ballZone, idLabel, isUnvalidated, matchShortLabel, similarEmptyState, similarHeadline } from "../lib/similar";
import { syntheticAnalysis } from "../test/fixtures";
import { SimilarPanel } from "./SimilarPanel";
import { WhyPanel } from "./WhyPanel";

const players = new Map<string, PlayerOut>([
  ["away_17", { player_id: "away_17", team: "away", jersey: 17 }],
  ["home_9", { player_id: "home_9", team: "home", jersey: 9 }],
]);

const ST: StatementOut[] = [
  { kind: "pressure", text: "Synthetic: away_17 has space.", metric: "pressure.x", value: 4.3, provenance: "derived", players: ["away_17", "home_9"] },
  { kind: "option", text: "Synthetic option line.", metric: "pass_options.y", value: 0.8, provenance: "estimated", players: [] },
  { kind: "caveat", text: "Synthetic caveat: a model, not a certainty.", metric: "model", value: null, provenance: "estimated", players: [] },
];

function similar(over: Partial<SimilarMomentsOut> = {}): SimilarMomentsOut {
  return {
    match_id: "metrica-sample-1",
    frame: 100,
    decision_frame: 104,
    passer_id: "away_17",
    k: 3,
    neighbours: [
      { match_id: "metrica-sample-2", frame: 18521, possession_id: null, team: "home", passer_id: "home_2", completed: false, distance: 4.2, ball_x_att: -37.9, press_nearest_def_m: 3.2, passer_control_s: 0 },
      { match_id: "metrica-sample-3", frame: 500, possession_id: null, team: "away", passer_id: "away_5", completed: true, distance: 4.9, ball_x_att: 12, press_nearest_def_m: null, passer_control_s: null },
      { match_id: "metrica-sample-3", frame: 900, possession_id: null, team: "away", passer_id: "away_6", completed: true, distance: 5.1, ball_x_att: 1, press_nearest_def_m: 1.1, passer_control_s: 0.4 },
    ],
    completed_share: 0.667,
    provider: "metrica",
    features: ["a", "b"],
    validation: null,
    disclaimer: "Synthetic disclaimer: a comparison, not a prediction.",
    ...over,
  };
}

describe("parse: explanations and temporal context", () => {
  it("accepts analyses with and without the new fields", () => {
    expect(parseAnalysis(syntheticAnalysis()).explanations).toBeUndefined();
    const a = parseAnalysis({ ...syntheticAnalysis(), temporal: { passer_control_s: 0.3, x: null }, explanations: ST });
    expect(a.explanations).toHaveLength(3);
    expect(parseAnalysis({ ...syntheticAnalysis(), temporal: null, explanations: [] }).temporal).toBeNull();
  });

  it("rejects malformed statements and temporal values", () => {
    expect(() => parseAnalysis({ ...syntheticAnalysis(), explanations: [{ ...ST[0], players: "away_17" }] })).toThrow(/players/);
    expect(() => parseAnalysis({ ...syntheticAnalysis(), explanations: [{ ...ST[0], text: "" }] })).toThrow(/text/);
    expect(() => parseAnalysis({ ...syntheticAnalysis(), explanations: [{ kind: "trend" }] })).toThrow(ContractError);
    expect(() => parseAnalysis({ ...syntheticAnalysis(), temporal: { x: "fast" } })).toThrow(/temporal.x/);
  });
});

describe("parse: similar moments", () => {
  it("accepts a response, normalising validation", () => {
    expect(parseSimilar(similar()).validation).toBeNull();
    const v = parseSimilar(similar({ validation: { ref: "experiments/results/x/results.json", knn_beats_base_rate: true, log_loss: { knn: 0.6 } } }));
    expect(v.validation?.knn_beats_base_rate).toBe(true);
    const partial = parseSimilar({ ...similar(), validation: { ref: null } });
    expect(partial.validation).toEqual({ ref: null, knn_beats_base_rate: null, log_loss: null });
  });

  it("rejects malformed neighbours, validation and a missing disclaimer", () => {
    const bad = similar();
    expect(() => parseSimilar({ ...bad, neighbours: [{ ...bad.neighbours[0], completed: "yes" }] })).toThrow(/completed/);
    expect(() => parseSimilar({ ...bad, neighbours: [{ ...bad.neighbours[0], distance: null }] })).toThrow(/distance/);
    expect(() => parseSimilar({ ...bad, disclaimer: "" })).toThrow(/disclaimer/);
    expect(() => parseSimilar({ ...bad, validation: { knn_beats_base_rate: "no" } })).toThrow(/knn_beats_base_rate/);
    expect(() => parseSimilar({ k: 1 })).toThrow(ContractError);
  });
});

describe("explain and similar helpers", () => {
  it("always returns the four groups in order, with unknown kinds kept before the caveats", () => {
    expect(groupStatements([]).map((g) => g.kind)).toEqual(["pressure", "trend", "option", "caveat"]);
    const g = groupStatements([...ST, { ...ST[1]!, kind: "future_kind", text: "x" }]);
    expect(g.map((x) => x.kind)).toEqual(["pressure", "trend", "option", "other", "caveat"]);
    expect(g.find((x) => x.kind === "other")?.items).toHaveLength(1);
  });

  it("words the headline, labels and validation", () => {
    expect(similarHeadline(similar())).toBe("In 3 similar situations from other matches, 2 passes were completed.");
    expect(similarHeadline(similar({ neighbours: [similar().neighbours[1]!] }))).toBe(
      "In 1 similar situation from other matches, 1 pass was completed.",
    );
    expect(isUnvalidated(similar())).toBe(true);
    expect(isUnvalidated(similar({ validation: { ref: "r", knn_beats_base_rate: false, log_loss: null } }))).toBe(true);
    expect(isUnvalidated(similar({ validation: { ref: "r", knn_beats_base_rate: true, log_loss: null } }))).toBe(false);
    expect(idLabel("home_2")).toBe("#2 Home");
    expect(idLabel("ref_1")).toBe("ref_1");
    expect(matchShortLabel("metrica-sample-2")).toBe("Metrica game 2");
    expect(matchShortLabel("skillcorner-1886347")).toBe("SkillCorner match 1886347");
    expect(ballZone(-37.9)).toBe("ball 38 m into own half");
    expect(ballZone(12)).toBe("ball 12 m into the opponent half");
    expect(ballZone(1)).toBe("ball near halfway");
    expect(ballZone(null)).toBeNull();
  });

  it("maps 404 and 503 to quiet empty states, everything else to errors", () => {
    expect(similarEmptyState(new ApiError(404, "no pass decision starts within 2 s"))).toEqual({ kind: "no-decision" });
    expect(similarEmptyState(new ApiError(503, "no similarity index"))).toEqual({ kind: "no-index" });
    expect(similarEmptyState(new ApiError(503, "Cannot reach", true))).toBeNull();
    expect(similarEmptyState(new ApiError(500, "boom"))).toBeNull();
    expect(similarEmptyState(new Error("x"))).toBeNull();
  });
});

describe("WhyPanel", () => {
  const analysis = syntheticAnalysis({ frame: 101, explanations: ST });

  it("groups statements, shows them as given with labels and provenance, and keeps caveats visible", () => {
    render(<WhyPanel analysis={analysis} frame={101} playing={false} loading={false} players={players} onTrace={() => {}} />);
    const items = screen.getAllByTestId("why-statement");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("Synthetic: #17 Away has space.");
    expect(items[0]).toHaveTextContent("derived");
    expect(within(screen.getByTestId("why-group-caveat")).getByText(/Synthetic caveat/)).toBeVisible();
    expect(screen.getByTestId("why-group-trend")).toHaveTextContent("No notable change over the last second.");
  });

  it("traces a statement's players on focus and hover, and clears on blur", async () => {
    const onTrace = vi.fn();
    render(<WhyPanel analysis={analysis} frame={101} playing={false} loading={false} players={players} onTrace={onTrace} />);
    const btn = screen.getByRole("button", { name: /#17 Away has space/ });
    act(() => btn.focus());
    expect(onTrace).toHaveBeenLastCalledWith(["away_17", "home_9"]);
    act(() => btn.blur());
    expect(onTrace).toHaveBeenLastCalledWith([]);
    await userEvent.hover(btn);
    expect(onTrace).toHaveBeenLastCalledWith(["away_17", "home_9"]);
    // statements with no players are plain text, not buttons
    expect(screen.queryByRole("button", { name: /Synthetic option line/ })).toBeNull();
  });

  it("marks a stale frame and handles a backend without explanations", () => {
    const { rerender } = render(
      <WhyPanel analysis={analysis} frame={105} playing={false} loading={true} players={players} onTrace={() => {}} />,
    );
    expect(screen.getByTestId("why-frame")).toHaveTextContent("showing frame 101");
    expect(screen.getByTestId("why-panel")).toHaveAttribute("aria-busy", "true");
    rerender(<WhyPanel analysis={syntheticAnalysis()} frame={101} playing={false} loading={false} players={players} onTrace={() => {}} />);
    expect(screen.getByTestId("why-unsupported")).toBeInTheDocument();
  });
});

describe("SimilarPanel", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches only on request and shows the headline, unvalidated badge, disclaimer and links", async () => {
    const spy = vi.spyOn(api, "similar").mockResolvedValue(similar());
    render(<SimilarPanel matchId="metrica-sample-1" frame={100} provider="Metrica Sports" />);
    expect(spy).not.toHaveBeenCalled();
    await userEvent.click(screen.getByTestId("similar-find"));
    expect(spy).toHaveBeenCalledWith("metrica-sample-1", 100, 8, expect.anything());
    expect(await screen.findByTestId("similar-headline")).toHaveTextContent(
      "In 3 similar situations from other matches, 2 passes were completed.",
    );
    expect(screen.getByTestId("similar-unvalidated")).toHaveTextContent("unvalidated");
    expect(screen.getByTestId("similar-disclaimer")).toHaveTextContent("Synthetic disclaimer");
    const first = screen.getAllByTestId("similar-item")[0]!;
    expect(first).toHaveTextContent("#2 Home");
    expect(first).toHaveTextContent("not completed");
    expect(first).toHaveTextContent("observed");
    expect(within(first).getByRole("link")).toHaveAttribute("href", "#/at/metrica-sample-2?f=18521");
  });

  it("shows no unvalidated badge when the held-out criterion is met", async () => {
    vi.spyOn(api, "similar").mockResolvedValue(similar({ validation: { ref: "r.json", knn_beats_base_rate: true, log_loss: null } }));
    render(<SimilarPanel matchId="metrica-sample-1" frame={100} provider="Metrica Sports" />);
    await userEvent.click(screen.getByTestId("similar-find"));
    await screen.findByTestId("similar-headline");
    expect(screen.queryByTestId("similar-unvalidated")).toBeNull();
  });

  it.each([
    [404, "no-decision", /No pass starts within 2 seconds/],
    [503, "no-index", /not available for Metrica Sports data yet/],
  ])("renders %s as a quiet empty state, not an error", async (status, kind, text) => {
    vi.spyOn(api, "similar").mockRejectedValue(new ApiError(status, "synthetic"));
    render(<SimilarPanel matchId="metrica-sample-1" frame={100} provider="Metrica Sports" />);
    await userEvent.click(screen.getByTestId("similar-find"));
    expect(await screen.findByTestId(`similar-empty-${kind}`)).toHaveTextContent(text);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("reports a real failure as an error with retry, and is disabled while playing", async () => {
    vi.spyOn(api, "similar").mockRejectedValue(new ApiError(500, "boom"));
    const { rerender } = render(<SimilarPanel matchId="metrica-sample-1" frame={100} provider="Metrica Sports" />);
    await userEvent.click(screen.getByTestId("similar-find"));
    expect(await screen.findByRole("alert")).toHaveTextContent("server hit an error");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    rerender(<SimilarPanel matchId="metrica-sample-1" frame={null} provider="Metrica Sports" />);
    expect(screen.getByTestId("similar-find")).toBeDisabled();
  });
});
