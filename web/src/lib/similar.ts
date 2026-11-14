import { ApiError } from "../api/client";
import type { SimilarMomentsOut } from "../api/types";

/** "#2 Home" for ids of the form home_2 / away_17 (players from other matches are not in this replay). */
export function idLabel(id: string): string {
  const m = /^(home|away)_(.+)$/.exec(id);
  return m ? `#${m[2]} ${m[1] === "home" ? "Home" : "Away"}` : id;
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
