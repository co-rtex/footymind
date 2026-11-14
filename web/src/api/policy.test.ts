/**
 * Response-policy contract (SYNTHETIC payloads). The backend adds `policy` to scenario responses
 * and GET /api/response-policies; older backends send neither, and both must keep working.
 */
import { describe, expect, it, vi } from "vitest";
import { syntheticAnalysis } from "../test/fixtures";
import { ApiError, createApi } from "./client";
import { ContractError, parsePolicyInfo, parseResponsePolicies, parseScenario } from "./parse";

const POLICY = {
  name: "press_carrier",
  description: "Synthetic: the nearest defender steps towards the carrier.",
  assumptions: ["synthetic assumption"],
  validated: false,
  validation_ref: null,
};

function scenario(extra: Record<string, unknown> = {}) {
  return {
    observed: syntheticAnalysis(),
    hypothetical: syntheticAnalysis(),
    feasibility: [],
    movements: [
      { player_id: "away_2", from_x: 1, from_y: 2, to_x: 3, to_y: 4, provenance: "simulated", reason: "synthetic rule" },
    ],
    option_deltas: [],
    summary_observed: {},
    summary_hypothetical: {},
    ensemble_size: 0,
    supported: true,
    warnings: [],
    assumptions: [],
    disclaimer: "synthetic",
    value_model: "option_model",
    ...extra,
  };
}

describe("parsePolicyInfo / parseResponsePolicies", () => {
  it("accepts a full policy and treats a missing validation_ref as null", () => {
    expect(parsePolicyInfo(POLICY)).toEqual({ ...POLICY, status: null, verdict: null });
    const noRef: Record<string, unknown> = { ...POLICY };
    delete noRef.validation_ref;
    expect(parsePolicyInfo(noRef).validation_ref).toBeNull();
    expect(parsePolicyInfo({ ...POLICY, validated: true, validation_ref: "C30" }).validation_ref).toBe("C30");
  });

  it("keeps the backend's status verdict, and tolerates it missing, null or blank (older backends)", () => {
    const verdict = "experimental: synthetic verdict (not a result)";
    expect(parsePolicyInfo({ ...POLICY, status: verdict }).status).toBe(verdict);
    expect(parsePolicyInfo({ ...POLICY, status: `  ${verdict} ` }).status).toBe(verdict);
    expect(parsePolicyInfo({ ...POLICY, status: null }).status).toBeNull();
    expect(parsePolicyInfo({ ...POLICY, status: "   " }).status).toBeNull();
    expect(parsePolicyInfo(POLICY).status).toBeNull();
    expect(() => parsePolicyInfo({ ...POLICY, status: 3 })).toThrow(/status/);
    expect(parseResponsePolicies([{ ...POLICY, status: verdict }])[0]?.status).toBe(verdict);
  });

  it("keeps a known machine-readable verdict and ignores an unknown one (SYNTHETIC values)", () => {
    expect(parsePolicyInfo({ ...POLICY, verdict: "not_supported" }).verdict).toBe("not_supported");
    expect(parsePolicyInfo({ ...POLICY, verdict: "observational_supported" }).verdict).toBe("observational_supported");
    expect(parsePolicyInfo({ ...POLICY, verdict: "some_future_verdict" }).verdict).toBeNull();
    expect(parsePolicyInfo({ ...POLICY, verdict: null }).verdict).toBeNull();
    expect(parsePolicyInfo(POLICY).verdict).toBeNull();
    expect(() => parsePolicyInfo({ ...POLICY, verdict: 1 })).toThrow(/verdict/);
  });

  it("rejects malformed policies", () => {
    expect(() => parsePolicyInfo({ ...POLICY, validated: "no" })).toThrow(/validated/);
    expect(() => parsePolicyInfo({ ...POLICY, assumptions: [1] })).toThrow(/assumptions/);
    expect(() => parsePolicyInfo({ ...POLICY, name: "" })).toThrow(/name/);
    expect(() => parsePolicyInfo({ ...POLICY, validation_ref: 3 })).toThrow(/validation_ref/);
    const noDesc: Record<string, unknown> = { ...POLICY };
    delete noDesc.description;
    expect(() => parsePolicyInfo(noDesc)).toThrow(ContractError);
  });

  it("parses the list, keeps unknown policy names and rejects duplicates", () => {
    const list = parseResponsePolicies([{ ...POLICY, name: "none" }, POLICY, { ...POLICY, name: "future_policy" }]);
    expect(list.map((p) => p.name)).toEqual(["none", "press_carrier", "future_policy"]);
    expect(() => parseResponsePolicies([POLICY, POLICY])).toThrow(/duplicate/);
    expect(() => parseResponsePolicies({})).toThrow(ContractError);
  });
});

describe("parseScenario with and without a policy", () => {
  it("accepts an older response with no policy field", () => {
    const s = parseScenario(scenario());
    expect(s.policy).toBeUndefined();
    expect(s.movements[0]?.provenance).toBe("simulated");
  });

  it("accepts a null policy and a well-formed policy", () => {
    expect(parseScenario(scenario({ policy: null })).policy).toBeNull();
    expect(parseScenario(scenario({ policy: POLICY })).policy).toEqual({ ...POLICY, status: null, verdict: null });
    const withStatus = { ...POLICY, status: "experimental: synthetic verdict" };
    expect(parseScenario(scenario({ policy: withStatus })).policy?.status).toBe("experimental: synthetic verdict");
  });

  it("rejects a malformed policy or movement", () => {
    expect(() => parseScenario(scenario({ policy: { name: "x" } }))).toThrow(/scenario.policy/);
    expect(() => parseScenario(scenario({ movements: [{ player_id: "a" }] }))).toThrow(/movement/);
    expect(() =>
      parseScenario(
        scenario({ movements: [{ player_id: "a", from_x: null, from_y: 0, to_x: 0, to_y: 0, provenance: "simulated" }] }),
      ),
    ).toThrow(/from_x/);
  });
});

describe("api client: response policies and unreachable API", () => {
  const json = (status: number, body: unknown) =>
    Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

  it("requests the policy list for a match", async () => {
    const fetchMock = vi.fn(() => json(200, [POLICY]));
    const api = createApi(fetchMock as unknown as typeof fetch);
    const list = await api.responsePolicies("metrica-sample-1");
    expect(fetchMock).toHaveBeenCalledWith("/api/response-policies?match_id=metrica-sample-1", expect.anything());
    expect(list[0]?.name).toBe("press_carrier");
  });

  it("reports 404 from an older backend as a plain ApiError", async () => {
    const api = createApi((() => json(404, { detail: "Not Found" })) as unknown as typeof fetch);
    await expect(api.responsePolicies("m")).rejects.toMatchObject({ status: 404, unreachable: false });
  });

  it("treats an empty-bodied 5xx (a proxy in front of a stopped API) as unreachable", async () => {
    for (const status of [500, 502, 503, 504]) {
      const api = createApi((() => Promise.resolve(new Response("", { status }))) as unknown as typeof fetch);
      const err = (await api.matches().catch((e: unknown) => e)) as ApiError;
      expect(err).toBeInstanceOf(ApiError);
      expect(err.unreachable).toBe(true);
      expect(err.message).toMatch(/Cannot reach the FootyMind API/);
    }
  });

  it("keeps a FastAPI 500 with a JSON detail as a server error, not unreachable", async () => {
    const api = createApi((() => json(500, { detail: "boom" })) as unknown as typeof fetch);
    await expect(api.matches()).rejects.toMatchObject({ status: 500, unreachable: false, detail: "boom" });
  });
});
