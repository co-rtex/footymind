import { describe, expect, it, vi } from "vitest";
import { syntheticAnalysis, syntheticReplay } from "../test/fixtures";
import { describeError } from "../lib/errors";
import { ApiError, createApi, formatErrorDetail, REQUEST_TIMEOUT_MS } from "./client";
import { ContractError, parseAnalysis, parsePossessions, parseReplay, parseScenario } from "./parse";

describe("parseReplay (synthetic fixture)", () => {
  it("converts to typed arrays with NaN for unobserved values", () => {
    const r = parseReplay(syntheticReplay());
    expect(r.nFrames).toBe(3);
    expect(r.nPlayers).toBe(2);
    expect(r.frame).toBeInstanceOf(Int32Array);
    expect(r.xy).toBeInstanceOf(Float32Array);
    expect(Array.from(r.frame)).toEqual([100, 101, 102]);
    // frame 1, player 1 is null -> NaN
    expect(Number.isNaN(r.xy[6])).toBe(true);
    expect(Number.isNaN(r.xy[7])).toBe(true);
    expect(r.xy[10]).toBe(12);
    // ball unobserved at frame 2
    expect(Number.isNaN(r.ball[4])).toBe(true);
    expect(r.attackSign.get(2)).toEqual({ home: -1, away: 1 });
  });

  it("rejects payloads whose flat arrays do not match the declared shape", () => {
    expect(() => parseReplay(syntheticReplay({ players_xy: [1, 2, 3] }))).toThrow(ContractError);
    expect(() => parseReplay(syntheticReplay({ ball_xy: [0, 0] }))).toThrow(/ball_xy/);
    expect(() => parseReplay(syntheticReplay({ time_s: [1] }))).toThrow(/time_s/);
  });

  it("rejects non-increasing frames and missing keys", () => {
    expect(() => parseReplay(syntheticReplay({ frame: [100, 100, 101] }))).toThrow(/increasing/);
    const rest: Record<string, unknown> = { ...syntheticReplay() };
    delete rest.players_xy;
    expect(() => parseReplay(rest)).toThrow(/players_xy/);
    expect(() => parseReplay(null)).toThrow(ContractError);
  });
});

describe("parseAnalysis / parseScenario / parsePossessions", () => {
  it("accepts a valid analysis, including a null attacking team and no pitch control", () => {
    const a = parseAnalysis(syntheticAnalysis({ attacking_team: null, pitch_control: null, pass_options: [] }));
    expect(a.attacking_team).toBeNull();
    expect(a.pitch_control).toBeNull();
  });

  it("rejects a pitch-control grid whose size is not nx*ny", () => {
    const bad = syntheticAnalysis();
    bad.pitch_control = { ...bad.pitch_control!, values: [0.5] };
    expect(() => parseAnalysis(bad)).toThrow(/nx\*ny/);
  });

  it("validates both sides of a scenario", () => {
    const ok = {
      observed: syntheticAnalysis(),
      hypothetical: syntheticAnalysis(),
      feasibility: [],
      movements: [],
      option_deltas: [],
      summary_observed: {},
      summary_hypothetical: {},
      ensemble_size: 0,
      supported: true,
      warnings: [],
      assumptions: [],
      disclaimer: "synthetic",
    };
    expect(parseScenario(ok).supported).toBe(true);
    expect(() => parseScenario({ ...ok, hypothetical: { frame: 1 } })).toThrow(ContractError);
  });

  it("requires possession arrays", () => {
    expect(() => parsePossessions({})).toThrow(ContractError);
    expect(() => parsePossessions([{ possession_id: "x" }])).toThrow(/team/);
  });
});

describe("api client", () => {
  const json = (status: number, body: unknown) =>
    Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

  it("hits relative /api paths and parses replays into typed arrays", async () => {
    const fetchMock = vi.fn(() => json(200, syntheticReplay()));
    const api = createApi(fetchMock as unknown as typeof fetch);
    const r = await api.possessionReplay("synthetic-1:p0000");
    expect(fetchMock).toHaveBeenCalledWith("/api/possessions/synthetic-1%3Ap0000/replay", expect.anything());
    expect(r.xy).toBeInstanceOf(Float32Array);
  });

  it("posts scenarios as JSON", async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) => json(422, { detail: "no possession" }));
    const api = createApi(fetchMock as unknown as typeof fetch);
    await expect(
      api.scenario({ match_id: "m", frame: 1, interventions: [{ player_id: "away_2", x: 1, y: 2 }] }),
    ).rejects.toMatchObject({ status: 422, detail: "no possession" });
    const init = fetchMock.mock.calls[0]?.[1];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toMatchObject({ frame: 1, interventions: [{ player_id: "away_2" }] });
  });

  it("surfaces FastAPI error details", async () => {
    const api = createApi((() => json(404, { detail: "unknown match 'x'" })) as unknown as typeof fetch);
    await expect(api.matches()).rejects.toThrow("HTTP 404: unknown match 'x'");
  });

  it("formats pydantic validation errors", () => {
    expect(
      formatErrorDetail({ detail: [{ loc: ["body", "interventions", 0, "x"], msg: "Input should be <= 60" }] }, "x"),
    ).toBe("body.interventions.0.x: Input should be <= 60");
    expect(formatErrorDetail("nope", "fallback")).toBe("fallback");
  });

  it("maps network failures to a friendly ApiError", async () => {
    const api = createApi((() => Promise.reject(new TypeError("Failed to fetch"))) as unknown as typeof fetch);
    const err = await api.matches().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
    expect((err as ApiError).message).toMatch(/backend/);
  });

  it("rethrows aborts unchanged", async () => {
    const abort = new DOMException("aborted", "AbortError");
    const api = createApi((() => Promise.reject(abort)) as unknown as typeof fetch);
    await expect(api.matches()).rejects.toBe(abort);
  });

  /** A fetch that never answers: it settles only when its signal aborts (like a hung API). */
  function hangingFetch() {
    const signals: AbortSignal[] = [];
    const impl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          const s = init?.signal;
          if (!s) return;
          signals.push(s);
          if (s.aborted) return reject(new DOMException("aborted", "AbortError"));
          s.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true });
        }),
    );
    return { impl, signals };
  }

  it("times out a hung API with a clear, retryable 'timed out' error", async () => {
    vi.useFakeTimers();
    try {
      const { impl } = hangingFetch();
      const api = createApi(impl as unknown as typeof fetch, 5_000);
      const p = api.matches().catch((e: unknown) => e);
      await vi.advanceTimersByTimeAsync(4_999);
      expect(impl).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      const err = (await p) as ApiError;
      expect(err).toBeInstanceOf(ApiError);
      expect(err.timedOut).toBe(true);
      expect(err.unreachable).toBe(true);
      expect(err.message).toMatch(/did not answer within 5 s \(timed out\)/);
      const view = describeError(err);
      expect(view.retryable).toBe(true);
      expect(view.message).toMatch(/timed out/);
    } finally {
      vi.useRealTimers();
    }
  });

  it("uses a 30 s timeout by default and still honours the caller's abort", async () => {
    expect(REQUEST_TIMEOUT_MS).toBe(30_000);
    const { impl, signals } = hangingFetch();
    const api = createApi(impl as unknown as typeof fetch);
    const ac = new AbortController();
    const p = api.matches(ac.signal).catch((e: unknown) => e);
    await Promise.resolve();
    expect(signals[0]?.aborted).toBe(false);
    ac.abort();
    const err = await p;
    expect(err).toBeInstanceOf(DOMException);
    expect((err as DOMException).name).toBe("AbortError");
    expect(signals[0]?.aborted).toBe(true);
  });

  it("passes an already-aborted caller signal straight through", async () => {
    const { impl, signals } = hangingFetch();
    const api = createApi(impl as unknown as typeof fetch);
    const ac = new AbortController();
    ac.abort();
    await expect(api.matches(ac.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(signals[0]?.aborted).toBe(true);
  });

  it("does not time out a response that arrives in time", async () => {
    vi.useFakeTimers();
    try {
      const api = createApi((() => json(200, [])) as unknown as typeof fetch, 1_000);
      await expect(api.matches()).resolves.toEqual([]);
      await vi.advanceTimersByTimeAsync(2_000);
    } finally {
      vi.useRealTimers();
    }
  });
});
