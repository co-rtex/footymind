import { useEffect, useMemo, useRef, type KeyboardEvent } from "react";
import type { ReplayData } from "../api/parse";
import type { MatchSummary, PlayerOut } from "../api/types";
import { playerLabel } from "../lib/format";
import { labelledPlayers, matchTeams } from "../lib/teams";
import type { Playback } from "../lib/playback";
import { clockAt, eventKind, eventLabel, indexOfFrame, periodAt } from "../lib/replayIndex";
import { Segmented } from "./Segmented";

interface Props {
  playback: Playback;
  replay: ReplayData;
  match: MatchSummary | null;
  playing: boolean;
  speed: number;
  disabled: boolean;
  disabledReason?: string;
  onSpeed: (s: number) => void;
}

const SPEEDS = [0.5, 1, 2] as const;

export function Transport({ playback, replay, match, playing, speed, disabled, disabledReason, onSpeed }: Props) {
  const rangeRef = useRef<HTMLInputElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const periodRef = useRef<HTMLSpanElement>(null);
  const frameRef = useRef<HTMLSpanElement>(null);
  const players: Map<string, PlayerOut> = useMemo(() => labelledPlayers(replay.players, matchTeams(match)), [replay, match]);
  const last = replay.nFrames - 1;

  // Imperative per-frame DOM updates: no React render while playing.
  useEffect(() => {
    const update = (t: number) => {
      const clock = clockAt(replay, t, match);
      const period = `P${periodAt(replay, t)}`;
      if (rangeRef.current) {
        rangeRef.current.value = String(t);
        rangeRef.current.setAttribute("aria-valuetext", `${period} ${clock}`);
        rangeRef.current.style.setProperty("--progress", `${(t / Math.max(1, last)) * 100}%`);
      }
      if (clockRef.current) clockRef.current.textContent = clock;
      if (periodRef.current) periodRef.current.textContent = period;
      if (frameRef.current) frameRef.current.textContent = `frame ${replay.frame[Math.round(t)] ?? ""}`;
    };
    update(playback.t);
    return playback.onTick(update);
  }, [playback, replay, match, last]);

  const markers = useMemo(
    () =>
      replay.events.map((e) => {
        const idx = indexOfFrame(replay.frame, e.start_frame);
        return {
          e,
          idx,
          kind: eventKind(e),
          pos: (idx / Math.max(1, last)) * 100,
          label: `${eventLabel(e)} by ${playerLabel(e.from_player_id, players)} at ${clockAt(replay, idx, match)}`,
        };
      }),
    [replay, last, players, match],
  );

  const win = useMemo(() => {
    const p = replay.possession;
    if (!p) return null;
    const a = indexOfFrame(replay.frame, p.start_frame);
    const b = indexOfFrame(replay.frame, p.end_frame);
    return { left: (a / Math.max(1, last)) * 100, width: ((b - a) / Math.max(1, last)) * 100 };
  }, [replay, last]);

  const onRangeKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") playback.step(e.shiftKey ? -Math.round(replay.fps) : -1);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") playback.step(e.shiftKey ? Math.round(replay.fps) : 1);
    else if (e.key === "Home") playback.step(-replay.nFrames);
    else if (e.key === "End") playback.step(replay.nFrames);
    else return;
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div className={`transport${disabled ? " is-disabled" : ""}`} data-testid="transport">
      <div className="transport-buttons">
        <button
          type="button"
          className="icon-btn"
          onClick={() => playback.step(-1)}
          disabled={disabled}
          aria-label="Step back one frame"
          title="Step back (Left arrow)"
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M6 4v12M16 4L8 10l8 6z" fill="currentColor" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
          </svg>
        </button>
        <button
          type="button"
          className="icon-btn icon-btn--primary"
          onClick={() => playback.toggle()}
          disabled={disabled}
          aria-label={playing ? "Pause" : "Play"}
          title={playing ? "Pause (Space)" : "Play (Space)"}
          data-testid="play-toggle"
        >
          {playing ? (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <rect x="5" y="4" width="3.5" height="12" rx="1" fill="currentColor" />
              <rect x="11.5" y="4" width="3.5" height="12" rx="1" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M6 4l10 6-10 6z" fill="currentColor" />
            </svg>
          )}
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={() => playback.step(1)}
          disabled={disabled}
          aria-label="Step forward one frame"
          title="Step forward (Right arrow)"
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M14 4v12M4 4l8 6-8 6z" fill="currentColor" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      <div className="clock" aria-live="off">
        <span className="clock-period" ref={periodRef} />
        <span className="clock-time" ref={clockRef} data-testid="clock" />
        <span className="clock-frame" ref={frameRef} />
      </div>

      <div className="timeline">
        <div className="timeline-markers" aria-label="Events in this window">
          {markers.map((m) => (
            <button
              key={m.e.event_id}
              type="button"
              className={`marker marker--${m.kind}`}
              style={{ left: `${m.pos}%` }}
              title={m.label}
              aria-label={`Jump to ${m.label}`}
              disabled={disabled}
              tabIndex={-1}
              onClick={() => playback.step(m.idx - Math.round(playback.t))}
            />
          ))}
        </div>
        <div className="timeline-track">
          {win ? (
            <span
              className="timeline-window"
              style={{ left: `${win.left}%`, width: `${win.width}%` }}
              title="Possession window (the replay is padded by 1 s either side)"
            />
          ) : null}
          <input
            ref={rangeRef}
            type="range"
            className="scrubber"
            min={0}
            max={last}
            step="any"
            defaultValue={0}
            disabled={disabled}
            aria-label="Replay position"
            title={disabled ? disabledReason : "Scrub (Left/Right to step, Shift for 1 s)"}
            data-testid="scrubber"
            onKeyDown={onRangeKey}
            onInput={(e) => {
              if (playback.playing) playback.pause();
              playback.seek(Number((e.target as HTMLInputElement).value));
            }}
          />
        </div>
      </div>

      <Segmented<string>
        label="Playback speed"
        hideLabel
        value={String(speed)}
        disabled={disabled}
        onChange={(v) => onSpeed(Number(v))}
        options={SPEEDS.map((s) => ({ value: String(s), label: `${s}×` }))}
        testId="speed"
      />
    </div>
  );
}
