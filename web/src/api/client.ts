/**
 * Typed HTTP client. Every call goes to the relative `/api` path: the Vite dev
 * proxy forwards it in development and FastAPI serves it in production.
 */
import {
  parseAnalysis,
  parseDefensiveShadow,
  parseMatches,
  parsePossessions,
  parseReplay,
  parseScenario,
  type ReplayData,
} from "./parse";
import type {
  DefensiveShadowOut,
  FrameAnalysisOut,
  MatchSummary,
  PossessionSummary,
  ScenarioRequest,
  ScenarioResponse,
  ValueModel,
} from "./types";

export const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(status ? `HTTP ${status}: ${detail}` : detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

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

export async function requestJson<T>(
  path: string,
  parse: (raw: unknown) => T,
  init: RequestInit = {},
  fetchImpl: Fetch = fetch,
): Promise<T> {
  let res: Response;
  try {
    res = await fetchImpl(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init.headers ?? {}) },
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "Cannot reach the FootyMind API. Is the backend running?");
  }
  const text = await res.text();
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

export function createApi(fetchImpl: Fetch = (...a) => fetch(...a)) {
  return {
    matches: (signal?: AbortSignal): Promise<MatchSummary[]> =>
      requestJson("/matches", parseMatches, { signal }, fetchImpl),

    possessions: (matchId: string, minPasses = 3, signal?: AbortSignal): Promise<PossessionSummary[]> =>
      requestJson(`/matches/${enc(matchId)}/possessions?min_passes=${minPasses}`, parsePossessions, { signal }, fetchImpl),

    possessionReplay: (possessionId: string, signal?: AbortSignal): Promise<ReplayData> =>
      requestJson(`/possessions/${enc(possessionId)}/replay`, parseReplay, { signal }, fetchImpl),

    frameAnalysis: (matchId: string, frame: number, signal?: AbortSignal): Promise<FrameAnalysisOut> =>
      requestJson(`/matches/${enc(matchId)}/frames/${frame}/analysis`, parseAnalysis, { signal }, fetchImpl),

    /** Experimental and slow (about 0.4 s per frame). Rejects with ApiError 422 when there is no ball or no possession. */
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
      ),

    scenario: (req: ScenarioRequest, signal?: AbortSignal): Promise<ScenarioResponse> =>
      requestJson(
        "/scenarios",
        parseScenario,
        { method: "POST", body: JSON.stringify(req), headers: { "Content-Type": "application/json" }, signal },
        fetchImpl,
      ),
  };
}

export type Api = ReturnType<typeof createApi>;
export const api: Api = createApi();
