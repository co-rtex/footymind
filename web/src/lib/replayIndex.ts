import type { EventOut, MatchSummary } from "../api/types";
import type { ReplayData } from "../api/parse";
import { fmtClock, matchClockSeconds, titleCase } from "./format";

/** First replay index whose frame is >= `frame` (clamped to the window). */
export function indexOfFrame(frames: Int32Array, frame: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  if (hi < 0) return 0;
  if (frame <= (frames[0] as number)) return 0;
  if (frame >= (frames[hi] as number)) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((frames[mid] as number) < frame) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export type EventKind = "pass" | "shot" | "lost" | "other";

export function eventKind(e: EventOut): EventKind {
  const t = e.type.toUpperCase();
  if (t === "PASS") return "pass";
  if (t === "SHOT") return "shot";
  if (t === "BALL LOST") return "lost";
  return "other";
}

export function eventLabel(e: EventOut): string {
  const sub = e.subtype ? ` (${titleCase(e.subtype)})` : "";
  return `${titleCase(e.type)}${sub}`;
}

export function clockAt(replay: ReplayData, idx: number, match: MatchSummary | null): string {
  const i = Math.max(0, Math.min(replay.nFrames - 1, Math.round(idx)));
  const t = replay.timeS[i] as number;
  const p = replay.period[i] as number;
  const secs = match ? matchClockSeconds(t, p, match.periods, match.fps) : t;
  return fmtClock(secs);
}

export function periodAt(replay: ReplayData, idx: number): number {
  const i = Math.max(0, Math.min(replay.nFrames - 1, Math.round(idx)));
  return replay.period[i] as number;
}
