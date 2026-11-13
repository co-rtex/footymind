import { describe, expect, it } from "vitest";
import { ContractError, parseReplay } from "../api/parse";
import type { MatchSummary } from "../api/types";
import { discStyle, styleKindFor } from "../render/styles";
import { syntheticReplay } from "../test/fixtures";
import { ESTIMATED, NOT_DRAWN, OBSERVED, sampleTrack, segmentEstimated } from "./interp";
import { groupByProvider, providerInfo, splitCommonName } from "./providers";

// SYNTHETIC broadcast-style detection mask for the 3-frame x 2-player fixture:
// player 0 is detected in f0, extrapolated in f1, detected in f2;
// player 1 is detected in f0, null (not drawn) in f1, extrapolated in f2.
const DETECTED = [true, true, false, false, true, false];

describe("parseReplay players_detected (synthetic)", () => {
  it("is null for optical tracking (field absent or null)", () => {
    expect(parseReplay(syntheticReplay()).detected).toBeNull();
    const r = parseReplay(syntheticReplay({ players_detected: null }));
    expect(r.detected).toBeNull();
    expect(r.nEstimated).toBe(0);
  });

  it("becomes a Uint8Array [frame][player] and counts drawn estimated positions", () => {
    const r = parseReplay(syntheticReplay({ players_detected: DETECTED }));
    expect(r.detected).toBeInstanceOf(Uint8Array);
    expect(Array.from(r.detected!)).toEqual([1, 1, 0, 0, 1, 0]);
    // f1/p0 and f2/p1 are extrapolated and drawn; f1/p1 is null so it does not count
    expect(r.nEstimated).toBe(2);
  });

  it("rejects a mask with the wrong shape or non-boolean values", () => {
    expect(() => parseReplay(syntheticReplay({ players_detected: [true, false] }))).toThrow(/players_detected has 2/);
    expect(() =>
      parseReplay(syntheticReplay({ players_detected: [1, 1, 0, 0, 1, 0] as unknown as boolean[] })),
    ).toThrow(ContractError);
  });
});

describe("sampleTrack provenance", () => {
  const r = parseReplay(syntheticReplay({ players_detected: DETECTED }));
  const at = (t: number, k: number) => sampleTrack(r.xy, r.nPlayers, r.nFrames, t, k, { x: 0, y: 0 }, r.period, r.detected);

  it("reports observed on detected frames and estimated on extrapolated ones", () => {
    expect(at(0, 0)).toBe(OBSERVED);
    expect(at(1, 0)).toBe(ESTIMATED);
    expect(at(2, 0)).toBe(OBSERVED);
  });

  it("marks an interpolated position estimated if either contributing frame is extrapolated", () => {
    expect(at(0.5, 0)).toBe(ESTIMATED); // observed f0 blended with estimated f1
    expect(at(1.5, 0)).toBe(ESTIMATED); // estimated f1 blended with observed f2
    expect(at(0.01, 0)).toBe(ESTIMATED);
  });

  it("does not let a zero-weight neighbour change provenance", () => {
    expect(at(0, 0)).toBe(OBSERVED); // f1 (estimated) has zero weight at t = 0
  });

  it("uses the nearest frame's flag next to a gap and never draws a null", () => {
    expect(at(0.3, 1)).toBe(OBSERVED); // f1 is null; nearest f0 detected
    expect(at(1.7, 1)).toBe(ESTIMATED); // nearest f2 extrapolated
    expect(at(1, 1)).toBe(NOT_DRAWN);
  });

  it("treats every drawn position as observed without a mask", () => {
    const m = parseReplay(syntheticReplay());
    expect(sampleTrack(m.xy, m.nPlayers, m.nFrames, 0.5, 0, { x: 0, y: 0 }, m.period, m.detected)).toBe(OBSERVED);
  });

  it("flags trail segments touching an extrapolated frame", () => {
    expect(segmentEstimated(r.detected, 2, 1, 0)).toBe(true);
    expect(segmentEstimated(r.detected, 2, 2, 0)).toBe(true);
    expect(segmentEstimated(null, 2, 1, 0)).toBe(false);
    expect(segmentEstimated(r.detected, 2, 0, 0)).toBe(false);
  });
});

describe("player style selection", () => {
  it("maps provenance to the four position styles", () => {
    expect(styleKindFor("observed")).toBe("observed");
    expect(styleKindFor("derived")).toBe("observed");
    expect(styleKindFor("estimated")).toBe("estimated");
    expect(styleKindFor("hypothetical")).toBe("hypothetical");
    expect(styleKindFor("simulated")).toBe("simulated");
  });

  it("keeps the four styles mutually distinct", () => {
    const o = discStyle("observed");
    const e = discStyle("estimated");
    const h = discStyle("hypothetical");
    const s = discStyle("simulated");
    expect(o.fillAlpha).toBe(1);
    expect(o.ring).toBe("outline");
    // estimated is hollow with a dashed ring in the TEAM colour
    expect(e.fillAlpha).toBeLessThan(0.2);
    expect(e).toMatchObject({ ring: "dashed", ringColour: "team", numberColour: "team" });
    // hypothetical: dashed violet; simulated: dotted white
    expect(h).toMatchObject({ ring: "dashed", ringColour: "accent" });
    expect(s).toMatchObject({ ring: "dotted", ringColour: "text" });
    const key = (d: ReturnType<typeof discStyle>) => `${d.ring}|${d.ringColour}|${d.fillAlpha}|${d.dash.join(",")}`;
    expect(new Set([o, e, h, s].map(key)).size).toBe(4);
  });
});

describe("providers", () => {
  const m = (id: string, provider: string, name: string): MatchSummary =>
    ({ match_id: id, source: { provider, name, url: `https://example.invalid/${provider}`, license_note: "" } }) as MatchSummary;

  it("groups matches by provider in order of first appearance", () => {
    const groups = groupByProvider([
      m("metrica-sample-1", "metrica", "Metrica 1"),
      m("skillcorner-1", "skillcorner", "SC 1"),
      m("metrica-sample-2", "metrica", "Metrica 2"),
      m("skillcorner-2", "skillcorner", "SC 2"),
    ]);
    expect(groups.map((g) => [g.info.label, g.matches.length])).toEqual([
      ["Metrica Sports", 2],
      ["SkillCorner", 2],
    ]);
  });

  it("splits a long shared source name into a series prefix and short labels", () => {
    expect(
      splitCommonName([
        "SkillCorner open data, A-League 2024/25 match 1874553",
        "SkillCorner open data, A-League 2024/25 match 1886347",
      ]),
    ).toEqual({ prefix: "SkillCorner open data, A-League 2024/25", labels: ["Match 1874553", "Match 1886347"] });
    expect(splitCommonName(["Metrica Sports Sample Game 1", "Metrica Sports Sample Game 2"])).toEqual({
      prefix: "Metrica Sports Sample",
      labels: ["Game 1", "Game 2"],
    });
    expect(splitCommonName(["Alpha", "Beta"])).toBeNull();
    expect(splitCommonName(["Only one"])).toBeNull();
  });

  it("credits the right provider and explains broadcast extrapolation", () => {
    expect(providerInfo({ provider: "skillcorner" }).attribution).toMatch(/SkillCorner/);
    expect(providerInfo({ provider: "skillcorner" }).kind).toMatch(/extrapolated/);
    expect(providerInfo({ provider: "metrica" }).attribution).toMatch(/Metrica Sports/);
    expect(providerInfo({ provider: "acme" }).label).toBe("acme");
  });
});
