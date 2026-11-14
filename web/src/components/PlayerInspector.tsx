import { useEffect, useRef, useState, type FormEvent } from "react";
import type { PlayerOut, ProvenanceLabel } from "../api/types";
import { INTERVENTION_BOUNDS } from "../api/types";
import { fmtMetres, teamLabel } from "../lib/format";
import { ProvBadge } from "./ProvBadge";

export interface InspectedPlayer {
  player: PlayerOut;
  x: number | null;
  y: number | null;
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
}

export function PlayerInspector({ info, playing, editable, onMove, onClose, focusMove = 0 }: Props) {
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

  return (
    <section className="card card--inspector" aria-labelledby="pi-title" data-testid="player-inspector">
      <div className="card-head">
        <h2 id="pi-title" className="h-card">
          <span className={`jersey jersey--${player.team}`}>{player.jersey}</span> {teamLabel(player.team)} #{player.jersey}
          {info.isGk ? <span className="muted small"> · goalkeeper</span> : null}
        </h2>
        <button type="button" className="icon-btn icon-btn--sm" onClick={onClose} aria-label="Close player card">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <dl className="kv">
        <dt>Player id</dt>
        <dd>
          <code>{player.player_id}</code>
        </dd>
        <dt>Position</dt>
        <dd>
          {playing ? (
            <span className="muted">pause to inspect</span>
          ) : info.x !== null && info.y !== null ? (
            <>
              ({info.x.toFixed(1)}, {info.y.toFixed(1)}) m <ProvBadge prov={info.positionProv} />
            </>
          ) : (
            <span className="muted">not observed at this frame</span>
          )}
        </dd>
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
        <form className="move-form" onSubmit={submit} aria-label={`Move ${teamLabel(player.team)} #${player.jersey}`}>
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
      ) : null}
    </section>
  );
}
