import { describe, expect, it, vi } from "vitest";
import { ApiError, createApi } from "../api/client";
import { ContractError, parseDefensiveShadow } from "../api/parse";
import type { DefenderShadowOut, DefensiveShadowOut } from "../api/types";
import {
  HALO_MAX_RADIUS_M,
  HALO_MIN_RADIUS_M,
  rankShadow,
  shadowDrawn,
  shadowHalo,
  shadowStanding,
  shadowStrength,
  suppressedReceiver,
} from "./shadow";

// SYNTHETIC fixture: invented values shaped like DefensiveShadowOut. Not real data.
function defender(id: string, sum: number, distinguishable: boolean, receiver: string | null = "away_9", delta = 0.1): DefenderShadowOut {
  return {
    player_id: id,
    x: 10,
    y: -5,
    shadow_best: 0,
    shadow_sum: sum,
    alt_radius_m: 3.5,
    alternatives: [
      [13.5, -5],
      [10, -1.5],
    ],
    most_suppressed_receiver: receiver,
    most_suppressed_delta: delta,
    distinguishable_from_noise: distinguishable,
    provenance: "estimated",
  };
}

function syntheticShadow(defenders: DefenderShadowOut[]): DefensiveShadowOut {
  return {
    match_id: "synthetic-1",
    frame: 101,
    attacking_team: "away",
    carrier_id: "away_7",
    value_model: "option_model",
    v_best_observed: 0.8,
    v_sum_observed: 5.1,
    defenders,
    assumptions: ["synthetic assumption"],
    status: "experimental: synthetic",
  };
}

describe("parseDefensiveShadow (synthetic fixture)", () => {
  it("accepts a well-formed payload", () => {
    const s = parseDefensiveShadow(syntheticShadow([defender("home_2", 0.2, true)]));
    expect(s.defenders[0]?.alternatives[1]).toEqual([10, -1.5]);
    expect(s.status).toMatch(/^experimental/);
  });

  it("rejects missing keys, bad alternatives and non-boolean noise flags", () => {
    const noDefenders: Record<string, unknown> = { ...syntheticShadow([]) };
    delete noDefenders.defenders;
    expect(() => parseDefensiveShadow(noDefenders)).toThrow(/defenders/);
    const badAlt = syntheticShadow([{ ...defender("home_2", 0.2, true), alternatives: [[1, 2, 3]] as unknown as [number, number][] }]);
    expect(() => parseDefensiveShadow(badAlt)).toThrow(/\[x, y\] pair/);
    const badFlag = syntheticShadow([{ ...defender("home_2", 0.2, true), distinguishable_from_noise: "yes" as unknown as boolean }]);
    expect(() => parseDefensiveShadow(badFlag)).toThrow(ContractError);
    const missing: Record<string, unknown> = { ...defender("home_2", 0.2, true) };
    delete missing.most_suppressed_receiver;
    expect(() => parseDefensiveShadow(syntheticShadow([missing as unknown as DefenderShadowOut]))).toThrow(
      /most_suppressed_receiver/,
    );
  });

  it("client builds the query and surfaces 422 (no ball / no possession) as ApiError", async () => {
    const fetchMock = vi.fn((_u: string, _i?: RequestInit) =>
      Promise.resolve(new Response(JSON.stringify({ detail: "ball not observed at this frame" }), { status: 422 })),
    );
    const api = createApi(fetchMock as unknown as typeof fetch);
    const err = await api.defensiveShadow("synthetic-1", 70000).catch((e: unknown) => e);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/matches/synthetic-1/frames/70000/defensive-shadow?horizon_s=1&value_model=option_model",
    );
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).detail).toBe("ball not observed at this frame");
  });
});

describe("shadow display helpers", () => {
  it("draws only distinguishable, suppressive defenders", () => {
    expect(shadowDrawn(defender("a", 0.27, true))).toBe(true);
    expect(shadowDrawn(defender("b", 0.27, false))).toBe(false); // below the noise floor
    expect(shadowDrawn(defender("c", -0.04, true))).toBe(false); // distinguishable but not suppressive
    expect(shadowHalo(defender("b", 0.27, false))).toBeNull();
  });

  it("scales the halo monotonically on an absolute, clamped square-root scale", () => {
    expect(shadowStrength(0)).toBe(0);
    expect(shadowStrength(-1)).toBe(0);
    expect(shadowStrength(Number.NaN)).toBe(0);
    expect(shadowStrength(0.3)).toBe(1);
    expect(shadowStrength(3)).toBe(1);
    expect(shadowStrength(0.075)).toBeCloseTo(0.5);
    const small = shadowHalo(defender("a", 0.02, true))!;
    const big = shadowHalo(defender("a", 0.27, true))!;
    const huge = shadowHalo(defender("a", 5, true))!;
    expect(small.radiusM).toBeGreaterThan(HALO_MIN_RADIUS_M);
    expect(big.radiusM).toBeGreaterThan(small.radiusM);
    expect(big.alpha).toBeGreaterThan(small.alpha);
    expect(huge.radiusM).toBe(HALO_MAX_RADIUS_M);
    expect(huge.alpha).toBeLessThanOrEqual(0.75);
  });

  it("ranks by shadow_sum and classifies standing", () => {
    const ranked = rankShadow([defender("n", 0.01, false), defender("neg", -0.04, true), defender("top", 0.27, true)]);
    expect(ranked.map((d) => d.player_id)).toEqual(["top", "n", "neg"]);
    expect(ranked.map(shadowStanding)).toEqual(["suppressive", "within-noise", "less-suppressive"]);
  });

  it("ignores a suppressed receiver whose delta is not positive", () => {
    expect(suppressedReceiver(defender("a", 0.2, true, "away_9", 0.17))).toBe("away_9");
    expect(suppressedReceiver(defender("a", 0.2, true, "away_9", 0))).toBeNull();
    expect(suppressedReceiver(defender("a", 0.2, true, null, 0.2))).toBeNull();
  });
});
