import { describe, expect, it } from "vitest";
import type { OptionDeltaOut, PlayerOut } from "../api/types";
import { ApiError } from "../api/client";
import { ContractError } from "../api/parse";
import { describeError } from "./errors";
import { humaniseIds } from "./format";
import {
  comparePolicies,
  FALLBACK_POLICIES,
  isExperimental,
  orderPolicies,
  policyBadge,
  policyFitRange,
  policyLabel,
  sensitivityVerdict,
  verdictText,
} from "./policies";

// SYNTHETIC deltas: values are made up to exercise the comparison logic.
function d(id: string, delta: number | null): OptionDeltaOut {
  return {
    receiver_id: id,
    observed_p: 0.5,
    hypothetical_p: delta === null ? null : 0.5 + delta,
    delta,
    observed_p10: null,
    observed_p90: null,
    hypothetical_p10: null,
    hypothetical_p90: null,
    delta_sign_agreement: 1,
  };
}

const players = new Map<string, PlayerOut>([
  ["home_7", { player_id: "home_7", team: "home", jersey: 7 }],
  ["home_9", { player_id: "home_9", team: "home", jersey: 9 }],
  ["home_4", { player_id: "home_4", team: "home", jersey: 4 }],
]);

describe("policy labels and flags", () => {
  it("labels known policies and makes unknown names readable", () => {
    expect(policyLabel("none")).toBe("Static (no response)");
    expect(policyLabel("marker_follows")).toBe("Marker follows");
    expect(policyLabel("learned")).toBe("Learned forecaster");
    expect(policyLabel("learned_innovation")).toBe("Learned reaction to attackers");
    expect(policyLabel("future_policy_v2")).toBe("Future policy v2");
  });

  it("marks unvalidated behaviour models experimental, never the static baseline", () => {
    expect(isExperimental({ name: "none", validated: false })).toBe(false);
    expect(isExperimental({ name: "lane_cover", validated: false })).toBe(true);
    expect(isExperimental({ name: "learned", validated: true })).toBe(false);
    expect(isExperimental(null)).toBe(false);
  });

  it("derives the verdict badge from status and validated, with a safe fallback", () => {
    const b = (name: string, validated: boolean, status?: string | null) => policyBadge({ name, validated, status });
    // older backends send no status: experimental, except the static baseline
    expect(b("marker_follows", false)).toBe("experimental");
    expect(b("learned", true)).toBe("experimental");
    expect(b("none", false)).toBeNull();
    expect(b("none", false, null)).toBeNull();
    // the name never decides: a "none" that claims a verdict is badged by it, and any name can say "no response"
    expect(b("anything", false, "no response simulated")).toBeNull();
    expect(b("none", false, "experimental: something new")).toBe("experimental");
    // "validated" text needs the validated flag; a negative verdict always wins
    expect(b("x", false, "validated: synthetic")).toBe("experimental");
    expect(b("x", true, "validated: synthetic")).toBe("forecaster");
    expect(b("x", true, "Validated (observational): synthetic")).toBe("observational");
    expect(b("x", true, "validated: but fails a synthetic placebo")).toBe("not supported");
    expect(b("x", false, "experimental: consistent with real defender reactions in an observational check")).toBe(
      "observational",
    );
    // "consistent with" alone, without an observational check, is not enough
    expect(b("x", false, "experimental: consistent with a synthetic intuition")).toBe("experimental");
    expect(b("x", false, "experimental: unsupported by the synthetic check")).toBe("not supported");
    expect(b("x", false, "experimental: undershoots real movement")).toBe("not supported");
    expect(b("x", false, "")).toBe("experimental");
    expect(policyBadge(null)).toBeNull();
  });

  it("reads a fitted range only from an extrapolation statement in the assumptions", () => {
    const fit = (assumptions: string[]) => policyFitRange({ assumptions });
    expect(fit(["Linear: 95% of synthetic innovations are below about 3.1 m in 0.5 s, so beyond is an extrapolation."])).toEqual({
      sharePct: 95,
      limitM: 3.1,
      horizonS: 0.5,
    });
    expect(fit(["95% of synthetic innovations are below about 3.1 m in 0.5 s."])).toBeNull();
    expect(fit(["An extrapolation, with no numbers."])).toBeNull();
    expect(policyFitRange(null)).toBeNull();
  });

  it("puts the static baseline first and falls back to the two original policies", () => {
    const list = orderPolicies([
      { ...FALLBACK_POLICIES[1]!, name: "zonal_shift" },
      FALLBACK_POLICIES[0]!,
    ]);
    expect(list.map((p) => p.name)).toEqual(["none", "zonal_shift"]);
    expect(FALLBACK_POLICIES.map((p) => p.name)).toEqual(["none", "marker_follows"]);
  });
});

describe("comparePolicies", () => {
  it("joins by receiver, measures the gap and counts direction flips", () => {
    const c = comparePolicies([d("home_7", 0.1), d("home_9", -0.04), d("home_4", 0.002)], [
      d("home_7", 0.08),
      d("home_9", 0.03),
      d("home_4", -0.003),
    ]);
    expect(c.nCompared).toBe(3);
    // home_9 flips; home_4 changes sign but both sides are negligible, so it is not a flip
    expect(c.nFlips).toBe(1);
    expect(c.rows.find((r) => r.receiver_id === "home_9")?.signFlip).toBe(true);
    expect(c.rows.find((r) => r.receiver_id === "home_4")?.signFlip).toBe(false);
    expect(c.maxAbsGap).toBeCloseTo(0.07);
    expect(c.rows[0]?.receiver_id).toBe("home_7");
    expect(c.topStatic).toBe("home_7");
    expect(sensitivityVerdict(c)).toBe("changes");
    expect(verdictText(c, "marker_follows", players)).toMatch(/1 of 3 pass options change direction/);
  });

  it("calls a same-direction difference of 0.05 or more a shift, and less the same", () => {
    const shift = comparePolicies([d("home_7", 0.2)], [d("home_7", 0.12)]);
    expect(sensitivityVerdict(shift)).toBe("shifts");
    expect(verdictText(shift, "zonal_shift", players)).toMatch(/differs by up to 0\.08/);
    const same = comparePolicies([d("home_7", 0.2)], [d("home_7", 0.19)]);
    expect(sensitivityVerdict(same)).toBe("same");
    expect(verdictText(same, "zonal_shift", players)).toMatch(/Little difference/);
  });

  it("reports a change of the largest option and tolerates missing values", () => {
    const c = comparePolicies([d("home_7", 0.2), d("home_9", 0.1)], [d("home_7", 0.05), d("home_9", 0.15), d("home_4", null)]);
    expect(c.topPolicy).toBe("home_9");
    expect(sensitivityVerdict(c)).toBe("changes");
    expect(verdictText(c, "press_carrier", players)).toMatch(/from #7 Home \(static\) to #9 Home \(Press the carrier\)/);
    const empty = comparePolicies([], []);
    expect(empty.nCompared).toBe(0);
    expect(verdictText(empty, "press_carrier", players)).toMatch(/No pass options/);
  });
});

describe("humaniseIds", () => {
  it("replaces known player ids in backend text and leaves unknown ones", () => {
    expect(humaniseIds("marker of home_9 follows 50% (home_99 unknown)", players)).toBe(
      "marker of #9 Home follows 50% (home_99 unknown)",
    );
  });
});

describe("describeError", () => {
  it("explains an unreachable server and offers retry", () => {
    const v = describeError(new ApiError(502, "Cannot reach the FootyMind API. Is the backend running?", true));
    expect(v.message).toMatch(/not responding/);
    expect(v.retryable).toBe(true);
  });

  it("explains a 404 as a stale or mistyped link, without retry", () => {
    const v = describeError(new ApiError(404, "unknown possession 'x'"), "this possession");
    expect(v.message).toMatch(/could not find this possession/);
    expect(v.technical).toBe("HTTP 404: unknown possession 'x'");
    expect(v.retryable).toBe(false);
    expect(v.notFound).toBe(true);
  });

  it("covers request errors, server errors and contract mismatches", () => {
    expect(describeError(new ApiError(422, "no possession at this frame")).retryable).toBe(false);
    expect(describeError(new ApiError(500, "boom")).message).toMatch(/server hit an error/);
    expect(describeError(new ContractError("x")).message).toMatch(/does not understand/);
    expect(describeError(new Error("plain")).message).toBe("plain");
  });
});
