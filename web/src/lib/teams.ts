/**
 * Team names and match titles.
 *
 * SkillCorner summaries carry real team names, competition and date; Metrica is anonymised
 * ("Home"/"Away", no competition or date) and must stay that way, so every helper falls back
 * to the neutral labels and never invents a name. Team colours stay keyed by home/away.
 */
import type { MatchSummary, PlayerOut, Team } from "../api/types";

export interface TeamNames {
  home: string;
  away: string;
}

export const DEFAULT_TEAMS: TeamNames = { home: "Home", away: "Away" };

/** A player record with its team's display name attached (frontend only). */
export type LabelledPlayer = PlayerOut & { team_label?: string };

export function matchTeams(match: Pick<MatchSummary, "home_label" | "away_label"> | null | undefined): TeamNames {
  const home = match?.home_label?.trim();
  const away = match?.away_label?.trim();
  return { home: home || DEFAULT_TEAMS.home, away: away || DEFAULT_TEAMS.away };
}

/** True when the match has real team names (not the anonymised Home/Away defaults). */
export function hasTeamNames(t: TeamNames): boolean {
  return t.home !== DEFAULT_TEAMS.home || t.away !== DEFAULT_TEAMS.away;
}

export function teamName(team: Team | null | undefined, t: TeamNames = DEFAULT_TEAMS): string {
  return team === "home" ? t.home : team === "away" ? t.away : "None";
}

/** Player map whose labels read "#17 Adelaide United" (or "#17 Away" when anonymised). */
export function labelledPlayers(players: PlayerOut[], t: TeamNames): Map<string, LabelledPlayer> {
  return new Map(players.map((p) => [p.player_id, { ...p, team_label: teamName(p.team, t) }]));
}

/**
 * "Brisbane FC vs Adelaide United · A-League, Round 5 · 2024-11-23". Missing parts are left
 * out; an anonymised match is titled by its source name ("Metrica Sports Sample Game 1").
 */
export function matchTitle(
  match: Pick<MatchSummary, "home_label" | "away_label" | "competition" | "date" | "source"> | null | undefined,
): string {
  if (!match) return "";
  const t = matchTeams(match);
  const head = hasTeamNames(t) ? `${t.home} vs ${t.away}` : match.source.name;
  return [head, match.competition, match.date].filter((s): s is string => !!s && s.trim().length > 0).join(" · ");
}

/** Second line for a compact match card: "A-League, Round 5 · 2024-11-23", or null. */
export function matchFacts(match: Pick<MatchSummary, "competition" | "date">): string | null {
  const parts = [match.competition, match.date].filter((s): s is string => !!s && s.trim().length > 0);
  return parts.length ? parts.join(" · ") : null;
}
