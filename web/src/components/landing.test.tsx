/**
 * Landing page on a partial setup: featured situations are shown only for matches the server has.
 * The match list is SYNTHETIC (made-up summaries under real match ids); the API is mocked and
 * returns nothing, so no number is rendered.
 */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from "vitest";
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

describe("match choice and the possession list", () => {
  const MATCHES = [...METRICA_ONLY, summary("skillcorner-2017461")];
  beforeEach(() => {
    resetFeaturedLive();
    vi.spyOn(api, "scenario").mockRejectedValue(new Error("synthetic: not served"));
    vi.spyOn(api, "defensiveShadow").mockRejectedValue(new Error("synthetic: not served"));
  });
  afterEach(() => vi.restoreAllMocks());

  it("the match cards are one radio group: one Tab stop, arrow keys move the selection", async () => {
    vi.spyOn(api, "possessions").mockImplementation(() => never());
    const onSelect = vi.fn();
    render(<Landing matches={MATCHES} match={MATCHES[1]!} onSelectMatch={onSelect} onOpenPossession={() => {}} />);
    const cards = within(screen.getByTestId("match-groups")).getAllByRole("radio");
    expect(cards.filter((c) => c.tabIndex === 0)).toEqual([screen.getByTestId("match-card-metrica-sample-2")]);
    cards[1]!.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenLastCalledWith("metrica-sample-3");
    expect(cards[2]).toHaveFocus();
    // across the provider groups, and wrapping around
    await userEvent.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenLastCalledWith("skillcorner-2017461");
    await userEvent.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenLastCalledWith("metrica-sample-1");
    await userEvent.keyboard("{End}");
    expect(onSelect).toHaveBeenLastCalledWith("skillcorner-2017461");
    expect(cards[3]).toHaveFocus();
  });

  it("a link to a match this server lacks says so instead of showing nothing", () => {
    vi.spyOn(api, "possessions").mockImplementation(() => never());
    render(
      <Landing
        matches={MATCHES}
        match={null}
        requestedMatchId="no-such-match"
        onSelectMatch={() => {}}
        onOpenPossession={() => {}}
      />,
    );
    expect(screen.getByTestId("unknown-match")).toHaveTextContent('This server has no match "no-such-match"');
    // the first card is still reachable with Tab
    expect(screen.getByTestId("match-card-metrica-sample-1").tabIndex).toBe(0);
  });

  it("back from the lab, the possession list is brought into view and its heading focused once loaded", async () => {
    vi.spyOn(api, "possessions").mockResolvedValue([]);
    const scroll = vi.fn();
    const original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = scroll;
    onTestFinished(() => {
      Element.prototype.scrollIntoView = original;
    });
    const onRevealed = vi.fn();
    render(
      <Landing
        matches={MATCHES}
        match={MATCHES[0]!}
        revealList
        onRevealed={onRevealed}
        onSelectMatch={() => {}}
        onOpenPossession={() => {}}
      />,
    );
    await waitFor(() => expect(onRevealed).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("heading", { name: "Possessions" })).toHaveFocus();
    expect(scroll).toHaveBeenCalled();
  });

  it("without revealList the landing keeps focus and scroll where they are", async () => {
    vi.spyOn(api, "possessions").mockResolvedValue([]);
    const onRevealed = vi.fn();
    render(<Landing matches={MATCHES} match={MATCHES[0]!} onRevealed={onRevealed} onSelectMatch={() => {}} onOpenPossession={() => {}} />);
    await screen.findByText("No possessions match these filters");
    expect(onRevealed).not.toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Possessions" })).not.toHaveFocus();
  });
});
