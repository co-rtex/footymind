/**
 * Landing page on a partial setup: featured situations are shown only for matches the server has.
 * The match list is SYNTHETIC (made-up summaries under real match ids); the API is mocked and
 * returns nothing, so no number is rendered.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { MatchSummary } from "../api/types";
import { availableFeatured, FEATURED } from "../lib/featured";
import { resetFeaturedLive } from "../lib/featuredLive";
import { SYNTHETIC_SOURCE } from "../test/fixtures";
import { Landing } from "./Landing";

function summary(match_id: string): MatchSummary {
  return {
    match_id,
    source: SYNTHETIC_SOURCE,
    fps: 25,
    n_frames: 1000,
    periods: [{ period: 1, first_frame: 1, last_frame: 900, duration_s: 36, home_attacks_positive_x: true }],
    n_possessions: 1,
    n_events: 1,
    pitch: { length: 105, width: 68 },
    home_label: "Home",
    away_label: "Away",
  };
}

const METRICA_ONLY = ["metrica-sample-1", "metrica-sample-2", "metrica-sample-3"].map(summary);
const never = <T,>() => new Promise<T>(() => {});

function renderLanding(matches: MatchSummary[]) {
  return render(<Landing matches={matches} match={null} onSelectMatch={() => {}} onOpenPossession={() => {}} />);
}

describe("featured situations follow GET /api/matches", () => {
  beforeEach(() => {
    resetFeaturedLive();
    // rejected at once so the serial request queue moves on (the cards then show their error state)
    vi.spyOn(api, "scenario").mockRejectedValue(new Error("synthetic: not served"));
    vi.spyOn(api, "defensiveShadow").mockRejectedValue(new Error("synthetic: not served"));
    vi.spyOn(api, "possessions").mockImplementation(() => never());
  });
  afterEach(() => vi.restoreAllMocks());

  it("keeps only situations whose match is listed, in curated order", () => {
    expect(availableFeatured(METRICA_ONLY.map((m) => m.match_id)).map((f) => f.id)).toEqual(["press", "response"]);
    expect(availableFeatured(FEATURED.map((f) => f.matchId))).toEqual([...FEATURED]);
    expect(availableFeatured([])).toEqual([]);
  });

  it("a Metrica-only setup shows no broadcast card and never requests its match", async () => {
    renderLanding(METRICA_ONLY);
    const section = screen.getByTestId("featured-section");
    expect(within(section).getByTestId("featured-link")).toBeInTheDocument();
    expect(within(section).getByTestId("featured-response-link")).toBeInTheDocument();
    expect(within(section).queryByTestId("featured-broadcast-link")).toBeNull();
    expect(section).not.toHaveTextContent("The same question on broadcast data");
    expect(screen.getByTestId("featured-missing")).toHaveTextContent(
      "One more featured situation needs data that is not loaded on this server (SkillCorner · broadcast tracking).",
    );
    // press: one scenario; response: static + marker_follows. The broadcast match is never asked for.
    await waitFor(() => expect(api.scenario).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(api.defensiveShadow).toHaveBeenCalledTimes(1));
    const requested = [
      ...vi.mocked(api.scenario).mock.calls.map(([req]) => req.match_id),
      ...vi.mocked(api.defensiveShadow).mock.calls.map(([id]) => id),
    ];
    expect(requested.every((id) => id.startsWith("metrica-"))).toBe(true);
    expect(screen.getByTestId("tour-link")).toHaveAttribute("href", expect.stringContaining("metrica-sample-1"));
  });

  it("with every featured match listed, shows all cards and no missing-data note", () => {
    renderLanding([...METRICA_ONLY, summary("skillcorner-1899585")]);
    expect(screen.getByTestId("featured-broadcast-link")).toBeInTheDocument();
    expect(screen.queryByTestId("featured-missing")).toBeNull();
  });

  it("with no featured match listed, hides the featured section and the tour rather than linking to a 404", () => {
    renderLanding([summary("skillcorner-2017461")]);
    expect(screen.queryByTestId("featured-section")).toBeNull();
    expect(screen.queryByTestId("tour-card")).toBeNull();
    expect(screen.queryByTestId("tour-link-hero")).toBeNull();
    expect(screen.getByTestId("match-groups")).toBeInTheDocument();
  });
});
