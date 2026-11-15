import { api, ApiError } from "../api/client";
import type { MatchSummary, SimilarMoment, SimilarMomentsOut } from "../api/types";
import { fmtClock, matchClockSeconds } from "./format";
import { DEFAULT_TEAMS, hasTeamNames, matchTeams, type TeamNames } from "./teams";

/**
 * "#2 Home" for ids of the form home_2 / away_17 (players from other matches are not in this
 * replay); with that match's team names, "#2 Brisbane FC".
 */
export function idLabel(id: string, teams: TeamNames = DEFAULT_TEAMS): string {
  const m = /^(home|away)_(.+)$/.exec(id);
  return m ? `#${m[2]} ${m[1] === "home" ? teams.home : teams.away}` : id;
}

/** Name of a neighbour's match: "Brisbane FC vs Adelaide United" when known, else the short id label. */
export function neighbourMatchLabel(id: string, summary: MatchSummary | undefined): string {
  if (summary) {
    const t = matchTeams(summary);
    if (hasTeamNames(t)) return `${t.home} vs ${t.away}`;
  }
  return matchShortLabel(id);
}

/**
 * Match clock of a neighbour pass, "P1 35:48". With the neighbour match's summary the clock is
 * exact (from the frame, as in the lab); otherwise it falls back to the provider minute
 * (0.1 min resolution), shown as a whole minute ("P1 35'") so it claims no false precision.
 */
export function neighbourClock(n: SimilarMoment, summary: MatchSummary | undefined): string | null {
  const period = n.period ?? null;
  if (period !== null && summary && summary.fps > 0 && summary.periods.some((p) => p.period === period)) {
    return `P${period} ${fmtClock(matchClockSeconds(n.frame / summary.fps, period, summary.periods, summary.fps))}`;
  }
  if (n.match_minute !== null && n.match_minute !== undefined && Number.isFinite(n.match_minute)) {
    return `${period !== null ? `P${period} ` : ""}${Math.floor(n.match_minute)}'`;
  }
  return null;
}

/**
 * Where a validation reference can be opened: absolute URLs as they are, repository paths under
 * VITE_REPO_BLOB_URL when the build sets it (for example a GitHub ".../blob/<branch>" URL), else null.
 */
export function refHref(
  ref: string | null | undefined,
  blobBase: string | undefined = import.meta.env.VITE_REPO_BLOB_URL as string | undefined,
): string | null {
  if (!ref) return null;
  if (/^https?:\/\//.test(ref)) return ref;
  if (blobBase) return `${blobBase.replace(/\/+$/, "")}/${ref.replace(/^\/+/, "")}`;
  return null;
}

/** Short, readable name for a match id. */
export function matchShortLabel(id: string): string {
  const metrica = /^metrica-sample-(\d+)$/.exec(id);
  if (metrica) return `Metrica game ${metrica[1]}`;
  const sc = /^skillcorner-(\d+)$/.exec(id);
  if (sc) return `SkillCorner match ${sc[1]}`;
  return id;
}

/** Ball position along the attacking direction in words (metres from halfway). */
export function ballZone(xAtt: number | null): string | null {
  if (xAtt === null || !Number.isFinite(xAtt)) return null;
  const d = Math.round(Math.abs(xAtt));
  if (d < 3) return "ball near halfway";
  return xAtt < 0 ? `ball ${d} m into own half` : `ball ${d} m into the opponent half`;
}

export function completedCount(s: SimilarMomentsOut): number {
  return s.neighbours.filter((n) => n.completed).length;
}

/** "In 8 similar situations from other matches, 3 passes were completed." */
export function similarHeadline(s: SimilarMomentsOut): string {
  const n = s.neighbours.length;
  const m = completedCount(s);
  return `In ${n} similar ${n === 1 ? "situation" : "situations"} from other matches, ${m} ${
    m === 1 ? "pass was" : "passes were"
  } completed.`;
}

/** The pre-registered held-out criterion is explicitly met. */
export function isValidated(s: SimilarMomentsOut): boolean {
  return s.validation?.knn_beats_base_rate === true;
}

/** Not validated unless the pre-registered held-out criterion is explicitly met. */
export function isUnvalidated(s: SimilarMomentsOut): boolean {
  return s.validation?.knn_beats_base_rate !== true;
}

/** Expected "nothing to show" outcomes, rendered as quiet empty states rather than errors. */
export type SimilarEmpty = { kind: "no-decision" } | { kind: "no-index" };

export function similarEmptyState(e: unknown): SimilarEmpty | null {
  if (e instanceof ApiError && !e.unreachable) {
    if (e.status === 404) return { kind: "no-decision" };
    if (e.status === 503) return { kind: "no-index" };
  }
  return null;
}

/** Match summaries for neighbour rows (team names, period starts), fetched once per session. */
let matchIndex: Promise<Map<string, MatchSummary>> | null = null;

export function loadMatchIndex(): Promise<Map<string, MatchSummary>> {
  if (!matchIndex) {
    matchIndex = api.matches().then(
      (list) => new Map(list.map((m) => [m.match_id, m])),
      () => {
        matchIndex = null; // rows fall back to ids and the provider minute; try again next time
        return new Map<string, MatchSummary>();
      },
    );
  }
  return matchIndex;
}

/** For tests: forget the cached match index. */
export function resetMatchIndex(): void {
  matchIndex = null;
}
