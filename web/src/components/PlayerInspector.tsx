import { useEffect, useRef, useState, type FormEvent } from "react";
import type { PlayerOut, ProvenanceLabel } from "../api/types";
import { INTERVENTION_BOUNDS } from "../api/types";
import { fmtMetres } from "../lib/format";
import { DEFAULT_TEAMS, teamName, type TeamNames } from "../lib/teams";
import { describeZone, type PitchZone } from "../lib/zones";
import { ProvBadge } from "./ProvBadge";

export interface InspectedPlayer {
  player: PlayerOut;
  x: number | null;
  y: number | null;
  /** football description of (x, y) from this player's team's point of view, or null */
  zone: PitchZone | null;
  speed: number | null;
  speedSource: string;
  positionProv: ProvenanceLabel;
  velocityProv: ProvenanceLabel;
  isGk: boolean | null;
}

interface Props {
  info: InspectedPlayer;
  playing: boolean;
  editable: boolean;
  onMove: (x: number, y: number) => void;
  onClose: () => void;
  /** incremented when the user picks this player from the Player menu to move them */
  focusMove?: number;
  teams?: TeamNames;
}

export function PlayerInspector({ info, playing, editable, onMove, onClose, focusMove = 0, teams = DEFAULT_TEAMS }: Props) {
  const { player } = info;
  const xRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (focusMove > 0) xRef.current?.focus();
  }, [focusMove]);
  const [mx, setMx] = useState("");
  const [my, setMy] = useState("");
  useEffect(() => {
    setMx(info.x !== null ? info.x.toFixed(1) : "");
    setMy(info.y !== null ? info.y.toFixed(1) : "");
  }, [info.x, info.y, player.player_id]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const x = Number(mx);
    const y = Number(my);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    onMove(Math.max(-INTERVENTION_BOUNDS.x, Math.min(INTERVENTION_BOUNDS.x, x)), Math.max(-INTERVENTION_BOUNDS.y, Math.min(INTERVENTION_BOUNDS.y, y)));
  };
  const known = info.x !== null && info.y !== null;

  return (
    <section className="card card--inspector" aria-labelledby="pi-title" data-testid="player-inspector">
      <div className="card-head">
        <h2 id="pi-title" className="h-card" tabIndex={-1}>
          <span className={`jersey jersey--${player.team}`} aria-hidden="true">
            {player.jersey}
          </span>{" "}
          #{player.jersey} {teamName(player.team, teams)}
          {info.isGk ? <span className="muted small"> · goalkeeper</span> : null}
        </h2>
        <button type="button" className="icon-btn icon-btn--sm" onClick={onClose} aria-label="Close player card">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <dl className="kv">
        <dt>Where</dt>
        <dd data-testid="player-zone">
          {playing ? (
            <span className="muted">pause to inspect</span>
          ) : known ? (
            <>
              {info.zone ? describeZone(info.zone) : <span className="muted">direction of play unknown</span>}{" "}
              <ProvBadge prov={info.positionProv} />
            </>
          ) : (
            <span className="muted">not observed at this frame</span>
          )}
        </dd>
        {!playing && known && info.zone ? (
          <>
            <dt>To goal</dt>
            <dd>
              {fmtMetres(info.zone.toGoal, 0)} <span className="muted small">from the centre of the goal they attack</span>
            </dd>
          </>
        ) : null}
        <dt>Speed</dt>
        <dd>
          {playing ? (
            <span className="muted">pause to inspect</span>
          ) : info.speed !== null ? (
            <>
              {info.speed.toFixed(1)} m/s <ProvBadge prov={info.velocityProv} />
              <span className="muted small"> {info.speedSource}</span>
            </>
          ) : (
            <span className="muted">unavailable</span>
          )}
        </dd>
      </dl>
      {editable && !playing && info.x !== null ? (
        <form className="move-form" onSubmit={submit} aria-label={`Move ${teamName(player.team, teams)} #${player.jersey}`}>
          <label>
            x (m)
            <input
              ref={xRef}
              type="number"
              step="0.1"
              min={-INTERVENTION_BOUNDS.x}
              max={INTERVENTION_BOUNDS.x}
              value={mx}
              onChange={(e) => setMx(e.target.value)}
              data-testid="move-x"
            />
          </label>
          <label>
            y (m)
            <input
              type="number"
              step="0.1"
              min={-INTERVENTION_BOUNDS.y}
              max={INTERVENTION_BOUNDS.y}
              value={my}
              onChange={(e) => setMy(e.target.value)}
              data-testid="move-y"
            />
          </label>
          <button type="submit" className="btn btn--sm" data-testid="move-submit">
            Move
          </button>
        </form>
      ) : null}
      {editable ? (
        <p className="small muted">
          Coordinates in metres from the centre spot: x along the pitch ({fmtMetres(52.5, 1)} to each goal line), y across it
          (positive is towards the top touchline).
        </p>
      ) : !playing && known ? (
        <details className="model-note">
          <summary>Pitch coordinates</summary>
          <p className="small muted tabular" data-testid="player-coords">
            x {info.x!.toFixed(1)} m, y {info.y!.toFixed(1)} m from the centre spot (x along the pitch, y towards the top
            touchline). Zones: thirds of the pitch; channels split at the six-yard box and the penalty area.
          </p>
        </details>
      ) : null}
    </section>
  );
}
