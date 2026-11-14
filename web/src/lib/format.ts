import type { PeriodInfo, PlayerOut, ProvenanceLabel, Team } from "../api/types";
import type { LabelledPlayer } from "./teams";

/** Football match clock: P1 counts from 0:00 and P2 from 45:00, P3 from 90:00 and so on. */
export function matchClockSeconds(timeS: number, period: number, periods: PeriodInfo[], fps: number): number {
  const p = periods.find((q) => q.period === period);
  if (!p) return timeS;
  const periodStart = p.first_frame / fps;
  const base = period <= 2 ? (period - 1) * 45 * 60 : 90 * 60 + (period - 3) * 15 * 60;
  return base + Math.max(0, timeS - periodStart);
}

export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function fmtPct(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "n/a";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "n/a";
  return p.toFixed(2);
}

export function fmtSigned(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  const s = Math.abs(v).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits); // never "-0.00" or "+0.00"
  return v > 0 ? `+${s}` : `−${s}`;
}

export function fmtMetres(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "n/a";
  return `${v.toFixed(digits)} m`;
}

export function teamLabel(t: Team | null | undefined): string {
  return t === "home" ? "Home" : t === "away" ? "Away" : "None";
}

export function titleCase(s: string | null | undefined): string {
  if (!s) return "";
  return s
    .toLowerCase()
    .split(/([ -])/)
    .map((w) => (w.length > 1 ? (w[0] as string).toUpperCase() + w.slice(1) : w))
    .join("");
}

export const PROVENANCE_TEXT: Record<ProvenanceLabel, string> = {
  observed: "Observed: measured by the tracking provider",
  derived: "Derived: computed deterministically from observed data",
  estimated: "Estimated: model output with uncalibrated literature parameters",
  hypothetical: "Hypothetical: position set by you in this scenario",
  simulated: "Simulated: moved by a defender response model (a rule or a learned model), not observed",
};

/**
 * "#10 Home" style label for a player id, falling back to the raw id. With a map built by
 * labelledPlayers() the team's display name is used ("#10 Brisbane FC").
 */
export function playerLabel(id: string | null | undefined, players: Map<string, PlayerOut>): string {
  if (!id) return "n/a";
  const p = players.get(id) as LabelledPlayer | undefined;
  return p ? `#${p.jersey} ${p.team_label ?? teamLabel(p.team)}` : id;
}

export function speedOf(vx: number | null, vy: number | null): number | null {
  if (vx === null || vy === null || !Number.isFinite(vx) || !Number.isFinite(vy)) return null;
  return Math.hypot(vx, vy);
}

/** Replace raw player ids (e.g. "away_18") inside backend text with "#18 Away" labels. */
export function humaniseIds(text: string, players: Map<string, PlayerOut>): string {
  return text.replace(/\b(?:home|away)_[A-Za-z0-9]+\b/g, (id) => (players.has(id) ? playerLabel(id, players) : id));
}
