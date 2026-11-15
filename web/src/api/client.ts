/**
 * Typed HTTP client. Every call goes to the relative `/api` path: the Vite dev
 * proxy forwards it in development and FastAPI serves it in production.
 */
import {
  parseAnalysis,
  parseDefensiveShadow,
  parseMatch,
  parseMatches,
  parsePossessions,
  parseReplay,
  parseResponsePolicies,
  parseScenario,
  parseSimilar,
  type ReplayData,
} from "./parse";
import type {
  DefensiveShadowOut,
  FrameAnalysisOut,
  MatchSummary,
  PolicyInfo,
  PossessionSummary,
  ScenarioRequest,
  ScenarioResponse,
  SimilarMomentsOut,
  ValueModel,
} from "./types";

export const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  /** true when the API itself could not be reached (network failure, or a dev-proxy error with no body) */
  readonly unreachable: boolean;
  /** true when the API did not answer within the request timeout (also `unreachable`) */
  readonly timedOut: boolean;
  constructor(status: number, detail: string, unreachable = false, timedOut = false) {
    super(status ? `HTTP ${status}: ${detail}` : detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.unreachable = unreachable || timedOut;
    this.timedOut = timedOut;
  }
}

const UNREACHABLE = "Cannot reach the FootyMind API. Is the backend running?";
/** Gateway statuses that a proxy (the Vite dev server, a reverse proxy) returns when the API is down. */
const GATEWAY = new Set([502, 503, 504]);

/** Turn FastAPI's `{detail: string | ValidationError[]}` into readable text. */
export function formatErrorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e) => {
          if (e && typeof e === "object") {
            const loc = Array.isArray((e as { loc?: unknown }).loc)
              ? ((e as { loc: unknown[] }).loc.join("."))
              : "";
            const msg = String((e as { msg?: unknown }).msg ?? "invalid");
            return loc ? `${loc}: ${msg}` : msg;
          }
          return String(e);
        })
        .join("; ");
    }
  }
  return fallback;
}

type Fetch = typeof fetch;

/**
 * A request (including reading its body) that takes longer than this is abandoned with a
 * "timed out" ApiError, so a hung API shows an error with Retry instead of an endless spinner.
 * The slowest endpoints (scenarios, the Defensive Shadow) answer in about a second.
 */
export const REQUEST_TIMEOUT_MS = 30_000;

function isAbort(e: unknown): boolean {
  return (
    (typeof DOMException !== "undefined" && e instanceof DOMException && e.name === "AbortError") ||
    (e instanceof Error && e.name === "AbortError")
  );
}

/**
 * The caller's signal combined with a timeout. Built by hand (not AbortSignal.any/timeout) so it
 * also works where those are missing, and so a timeout can be told apart from the caller's abort.
 */
function withTimeout(signal: AbortSignal | null | undefined, ms: number) {
  const ctrl = new AbortController();
  let timedOut = false;
  const onAbort = () => ctrl.abort(signal?.reason);
  if (signal?.aborted) ctrl.abort(signal.reason);
  else signal?.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, ms);
  return {
    signal: ctrl.signal,
    timedOut: () => timedOut,
    done: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

function timeoutError(ms: number): ApiError {
  return new ApiError(
    0,
    `The FootyMind API did not answer within ${Math.round(ms / 1000)} s (timed out). It may be overloaded or stuck.`,
    true,
    true,
  );
}

export async function requestJson<T>(
  path: string,
  parse: (raw: unknown) => T,
  init: RequestInit = {},
  fetchImpl: Fetch = fetch,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const t = withTimeout(init.signal, timeoutMs);
  let res: Response;
  let text: string;
  try {
    try {
      res = await fetchImpl(`${API_BASE}${path}`, {
        ...init,
        signal: t.signal,
        headers: { Accept: "application/json", ...(init.headers ?? {}) },
      });
    } catch (e) {
      if (t.timedOut()) throw timeoutError(timeoutMs);
      if (isAbort(e)) throw e;
      throw new ApiError(0, UNREACHABLE, true);
    }
    try {
      text = await res.text();
    } catch (e) {
      if (t.timedOut()) throw timeoutError(timeoutMs);
      if (isAbort(e)) throw e;
      throw new ApiError(res.status, UNREACHABLE, true);
    }
  } finally {
    t.done();
  }
  // A proxy in front of a stopped API answers 5xx with an empty body; FastAPI always sends JSON.
  if ((GATEWAY.has(res.status) || res.status === 500) && !text.trim()) {
    throw new ApiError(res.status, UNREACHABLE, true);
  }
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      if (!res.ok) throw new ApiError(res.status, text.slice(0, 200) || res.statusText);
      throw new ApiError(res.status, "Response was not valid JSON");
    }
  }
  if (!res.ok) throw new ApiError(res.status, formatErrorDetail(body, res.statusText || "Request failed"));
  return parse(body);
}

const enc = encodeURIComponent;

export function createApi(fetchImpl: Fetch = (...a) => fetch(...a), timeoutMs: number = REQUEST_TIMEOUT_MS) {
  return {
    matches: (signal?: AbortSignal): Promise<MatchSummary[]> =>
      requestJson("/matches", parseMatches, { signal }, fetchImpl, timeoutMs),

    /** One match summary (fast); the lab uses it so it never waits for the full match list. */
    match: (matchId: string, signal?: AbortSignal): Promise<MatchSummary> =>
      requestJson(`/matches/${enc(matchId)}`, parseMatch, { signal }, fetchImpl, timeoutMs),

    possessions: (matchId: string, minPasses = 3, signal?: AbortSignal): Promise<PossessionSummary[]> =>
      requestJson(`/matches/${enc(matchId)}/possessions?min_passes=${minPasses}`, parsePossessions, { signal }, fetchImpl, timeoutMs),

    framePossession: (matchId: string, frame: number, signal?: AbortSignal): Promise<PossessionSummary> =>
      requestJson(
        `/matches/${enc(matchId)}/frames/${frame}/possession`,
        (raw: unknown) => parsePossessions([raw])[0] as PossessionSummary,
        { signal },
        fetchImpl,
        timeoutMs,
      ),

    possessionReplay: (possessionId: string, signal?: AbortSignal): Promise<ReplayData> =>
      requestJson(`/possessions/${enc(possessionId)}/replay`, parseReplay, { signal }, fetchImpl, timeoutMs),

    frameAnalysis: (matchId: string, frame: number, signal?: AbortSignal): Promise<FrameAnalysisOut> =>
      requestJson(`/matches/${enc(matchId)}/frames/${frame}/analysis`, parseAnalysis, { signal }, fetchImpl, timeoutMs),

    /** Experimental and slow (under a second per frame). Rejects with ApiError 422 when there is no ball or no possession. */
    defensiveShadow: (
      matchId: string,
      frame: number,
      opts: { horizonS?: number; valueModel?: ValueModel } = {},
      signal?: AbortSignal,
    ): Promise<DefensiveShadowOut> =>
      requestJson(
        `/matches/${enc(matchId)}/frames/${frame}/defensive-shadow?horizon_s=${opts.horizonS ?? 1.0}&value_model=${
          opts.valueModel ?? "option_model"
        }`,
        parseDefensiveShadow,
        { signal },
        fetchImpl,
        timeoutMs,
      ),

    /**
     * Past pass decisions (other matches, same provider) most similar to the one starting near
     * this frame. Rejects with ApiError 404 (no pass decision within 2 s) or 503 (no index).
     */
    similar: (matchId: string, frame: number, k = 8, signal?: AbortSignal): Promise<SimilarMomentsOut> =>
      requestJson(`/matches/${enc(matchId)}/frames/${frame}/similar?k=${k}`, parseSimilar, { signal }, fetchImpl, timeoutMs),

    /**
     * Defender response policies the backend can simulate. Older backends do not have the
     * endpoint (404); callers fall back to FALLBACK_POLICIES (src/lib/policies.ts).
     */
    responsePolicies: (matchId: string, signal?: AbortSignal): Promise<PolicyInfo[]> =>
      requestJson(`/response-policies?match_id=${enc(matchId)}`, parseResponsePolicies, { signal }, fetchImpl, timeoutMs),

    scenario: (req: ScenarioRequest, signal?: AbortSignal): Promise<ScenarioResponse> =>
      requestJson(
        "/scenarios",
        parseScenario,
        { method: "POST", body: JSON.stringify(req), headers: { "Content-Type": "application/json" }, signal },
        fetchImpl,
        timeoutMs,
      ),
  };
}

export type Api = ReturnType<typeof createApi>;
export const api: Api = createApi();
