/**
 * Team names, match titles and neighbour clocks. SYNTHETIC summaries: the names below are
 * made-up placeholders, not real fixtures.
 */
import { describe, expect, it } from "vitest";
import { parseMatch, parseSimilar } from "../api/parse";
import type { MatchSummary, SimilarMoment, SimilarMomentsOut } from "../api/types";
import { playerLabel } from "./format";
import { idLabel, isUnvalidated, isValidated, neighbourClock, neighbourMatchLabel, refHref } from "./similar";
import { hasTeamNames, labelledPlayers, matchFacts, matchTeams, matchTitle, teamName } from "./teams";

function summary(over: Partial<MatchSummary> = {}): MatchSummary {
  return {
    match_id: "skillcorner-1",
    source: { provider: "skillcorner", name: "Synthetic source 1", url: "https://example.invalid", license_note: "synthetic" },
    fps: 10,
    n_frames: 1000,
    periods: [
      { period: 1, first_frame: 10, last_frame: 29760, duration_s: 2975, home_attacks_positive_x: true },
      { period: 2, first_frame: 29770, last_frame: 60529, duration_s: 3076, home_attacks_positive_x: false },
    ],
    n_possessions: 10,
    n_events: 100,
    pitch: { length: 105, width: 68 },
    home_label: "Synthetic Home FC",
    away_label: "Synthetic Away United",
    competition: "Synthetic League, Round 5",
    date: "2024-11-23",
    ...over,
  };
}

const ANON = summary({ match_id: "metrica-sample-1", home_label: "Home", away_label: "Away", competition: null, date: null, source: { provider: "metrica", name: "Metrica Sports Sample Game 1", url: "https://example.invalid", license_note: "synthetic" } });

describe("team names and match titles", () => {
  it("uses real names when present and the anonymised defaults otherwise", () => {
    expect(matchTeams(summary())).toEqual({ home: "Synthetic Home FC", away: "Synthetic Away United" });
    expect(matchTeams(ANON)).toEqual({ home: "Home", away: "Away" });
    expect(matchTeams(null)).toEqual({ home: "Home", away: "Away" });
    expect(matchTeams(summary({ home_label: "  ", away_label: "" }))).toEqual({ home: "Home", away: "Away" });
    expect(hasTeamNames(matchTeams(summary()))).toBe(true);
    expect(hasTeamNames(matchTeams(ANON))).toBe(false);
    expect(teamName("away", matchTeams(summary()))).toBe("Synthetic Away United");
    expect(teamName(null)).toBe("None");
  });

  it("builds the title and drops missing parts cleanly", () => {
    expect(matchTitle(summary())).toBe("Synthetic Home FC vs Synthetic Away United · Synthetic League, Round 5 · 2024-11-23");
    expect(matchTitle(summary({ competition: null }))).toBe("Synthetic Home FC vs Synthetic Away United · 2024-11-23");
    expect(matchTitle(summary({ competition: undefined, date: undefined }))).toBe("Synthetic Home FC vs Synthetic Away United");
    // anonymised: never invent names; the source name identifies the match
    expect(matchTitle(ANON)).toBe("Metrica Sports Sample Game 1");
    expect(matchTitle(null)).toBe("");
    expect(matchFacts(summary())).toBe("Synthetic League, Round 5 · 2024-11-23");
    expect(matchFacts(ANON)).toBeNull();
  });

  it("labels players with their team name, keeping the #N form", () => {
    const players = labelledPlayers([{ player_id: "away_17", team: "away", jersey: 17 }], matchTeams(summary()));
    expect(playerLabel("away_17", players)).toBe("#17 Synthetic Away United");
    const anon = labelledPlayers([{ player_id: "away_17", team: "away", jersey: 17 }], matchTeams(ANON));
    expect(playerLabel("away_17", anon)).toBe("#17 Away");
  });

  it("checks the optional match facts in the contract", () => {
    expect(parseMatch(summary()).competition).toBe("Synthetic League, Round 5");
    const old: Record<string, unknown> = { ...summary() };
    delete old.competition;
    delete old.date;
    expect(parseMatch(old).date).toBeUndefined();
    expect(() => parseMatch({ ...summary(), date: 20241123 })).toThrow(/date/);
    expect(() => parseMatch({ ...summary(), home_label: 3 })).toThrow(/home_label/);
  });
});

function nb(over: Partial<SimilarMoment> = {}): SimilarMoment {
  return {
    match_id: "skillcorner-1",
    frame: 29770 + 10 * 125.4,
    possession_id: null,
    team: "home",
    passer_id: "home_2",
    completed: true,
    distance: 4,
    ball_x_att: null,
    press_nearest_def_m: null,
    passer_control_s: null,
    period: 2,
    match_minute: 101.2,
    ...over,
  };
}

describe("neighbour rows", () => {
  it("derives an exact clock from the frame when the match summary is known", () => {
    // P2 starts at 45:00; 125.4 s into the period
    expect(neighbourClock(nb({ frame: 29770 + 1254 }), summary())).toBe("P2 47:05");
    expect(neighbourClock(nb({ period: 1, frame: 10 + 21483 }), summary())).toBe("P1 35:48");
  });

  it("falls back to the whole provider minute, or nothing", () => {
    expect(neighbourClock(nb(), undefined)).toBe("P2 101'");
    expect(neighbourClock(nb({ period: null }), undefined)).toBe("101'");
    expect(neighbourClock(nb({ period: undefined, match_minute: undefined }), undefined)).toBeNull();
    // a period the summary does not know falls back too
    expect(neighbourClock(nb({ period: 3 }), summary())).toBe("P3 101'");
  });

  it("names the neighbour's team and match when known", () => {
    expect(idLabel("home_2", matchTeams(summary()))).toBe("#2 Synthetic Home FC");
    expect(idLabel("away_9")).toBe("#9 Away");
    expect(neighbourMatchLabel("skillcorner-1", summary())).toBe("Synthetic Home FC vs Synthetic Away United");
    expect(neighbourMatchLabel("metrica-sample-2", ANON)).toBe("Metrica game 2");
    expect(neighbourMatchLabel("skillcorner-77", undefined)).toBe("SkillCorner match 77");
  });

  it("parses the optional clock fields", () => {
    const base = {
      match_id: "m",
      frame: 1,
      decision_frame: 1,
      passer_id: "home_1",
      k: 1,
      neighbours: [nb()],
      completed_share: 1,
      provider: "skillcorner",
      features: [],
      validation: null,
      disclaimer: "synthetic",
    };
    expect(parseSimilar(base).neighbours[0]?.period).toBe(2);
    expect(() => parseSimilar({ ...base, neighbours: [nb({ period: 1.5 })] })).toThrow(/period/);
    expect(() => parseSimilar({ ...base, neighbours: [{ ...nb(), match_minute: "35" }] })).toThrow(/match_minute/);
  });

  it("treats only an explicit true criterion as validated and links refs only when resolvable", () => {
    const s = (v: SimilarMomentsOut["validation"]) => ({ validation: v }) as SimilarMomentsOut;
    expect(isValidated(s({ ref: "r", knn_beats_base_rate: true, log_loss: null }))).toBe(true);
    expect(isUnvalidated(s({ ref: "r", knn_beats_base_rate: true, log_loss: null }))).toBe(false);
    expect(isValidated(s({ ref: "r", knn_beats_base_rate: null, log_loss: null }))).toBe(false);
    expect(isValidated(s(null))).toBe(false);
    expect(refHref("experiments/results/x/results.json", undefined)).toBeNull();
    expect(refHref("experiments/results/x/results.json", "https://example.invalid/blob/main/")).toBe(
      "https://example.invalid/blob/main/experiments/results/x/results.json",
    );
    expect(refHref("https://example.invalid/r.json", undefined)).toBe("https://example.invalid/r.json");
    expect(refHref(null)).toBeNull();
  });
});
