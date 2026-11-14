import { memo, useLayoutEffect, useRef, useState } from "react";
import type { FrameAnalysisOut, PlayerOut, StatementOut } from "../api/types";
import { humaniseIds, playerLabel } from "../lib/format";
import { groupStatements } from "../lib/explain";
import { ProvBadge } from "./ProvBadge";

interface Props {
  /** the latest analysis (may be for an earlier frame while the current one loads) */
  analysis: FrameAnalysisOut | null;
  /** the replay frame on screen */
  frame: number;
  playing: boolean;
  loading: boolean;
  players: Map<string, PlayerOut>;
  /** players to ring on the pitch (empty = none) */
  onTrace: (ids: string[]) => void;
}

function Statement({
  s,
  players,
  onTrace,
  pinned,
  onPin,
}: {
  s: StatementOut;
  players: Map<string, PlayerOut>;
  onTrace: (ids: string[]) => void;
  pinned: boolean;
  onPin: () => void;
}) {
  const text = humaniseIds(s.text, players);
  const traceable = s.players.length > 0;
  const who = s.players.map((id) => playerLabel(id, players)).join(", ");
  return (
    <li className="why-item" data-testid="why-statement" data-kind={s.kind} data-metric={s.metric}>
      {traceable ? (
        <button
          type="button"
          className={`why-text why-text--trace${pinned ? " is-pinned" : ""}`}
          aria-pressed={pinned}
          title={`Highlights ${who} on the pitch`}
          onMouseEnter={() => onTrace(s.players)}
          onMouseLeave={() => !pinned && onTrace([])}
          onFocus={() => onTrace(s.players)}
          onBlur={() => !pinned && onTrace([])}
          onClick={onPin}
        >
          {text}
          <span className="sr-only"> (highlights {who} on the pitch)</span>
        </button>
      ) : (
        <span className="why-text">{text}</span>
      )}
      <ProvBadge prov={s.provenance} />
    </li>
  );
}

/**
 * "Why this moment": the backend's evidence-grounded statements for the frame, grouped by kind
 * and shown verbatim (player ids rewritten as "#18 Away"). Focusing or hovering a statement
 * rings the players it names on the pitch. Caveats are always shown.
 */
export const WhyPanel = memo(function WhyPanel({ analysis, frame, playing, loading, players, onTrace }: Props) {
  const [pinned, setPinned] = useState<string | null>(null);
  // The number of statements varies by frame. The panel only ever grows (until the width
  // changes), so stepping through frames never pulls the panels below it up and down.
  const ref = useRef<HTMLElement>(null);
  const tallest = useRef({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const width = el.clientWidth;
    if (width !== tallest.current.width) {
      tallest.current = { width, height: 0 };
      el.style.minHeight = "";
    }
    const h = el.offsetHeight;
    if (h > tallest.current.height) {
      tallest.current.height = h;
      el.style.minHeight = `${h}px`;
    }
  });
  const statements = analysis?.explanations;
  const stale = !!analysis && (playing || analysis.frame !== frame);
  const groups = groupStatements(statements ?? []);
  const pin = (key: string, ids: string[]) => {
    if (pinned === key) {
      setPinned(null);
      onTrace([]);
    } else {
      setPinned(key);
      onTrace(ids);
    }
  };
  return (
    <section
      ref={ref}
      className={`card card--why${stale ? " is-stale" : ""}`}
      aria-labelledby="why-title"
      aria-busy={loading}
      data-testid="why-panel"
      data-frame={analysis?.frame}
    >
      <div className="card-head">
        <h2 id="why-title" className="h-card">
          Why this moment
        </h2>
        <span className="muted small tabular" data-testid="why-frame">
          {analysis ? (stale ? `showing frame ${analysis.frame}` : `frame ${analysis.frame}`) : ""}
          {loading ? <span className="spinner spinner--inline" aria-label="Updating" /> : null}
        </span>
      </div>
      {!analysis ? (
        <p className="muted small">Pause on a frame to see what stands out in it.</p>
      ) : statements === undefined ? (
        <p className="muted small" data-testid="why-unsupported">
          This server does not provide explanations yet.
        </p>
      ) : (
        <>
          <p className="small muted">
            Each line is generated from one measured or modelled quantity. Point at or tab to a line to highlight the
            players it mentions.
          </p>
          {groups.map((g) => (
            <div key={g.kind} className={`why-group why-group--${g.kind}`} data-testid={`why-group-${g.kind}`}>
              <h3 className="h-sub">{g.title}</h3>
              {g.items.length === 0 ? (
                <p className="small muted why-none">{g.empty}</p>
              ) : (
                <ul className="why-list">
                  {g.items.map((s, i) => {
                    // the pin is keyed by frame too, so it never outlives the statements it was made on
                    const key = `${analysis.frame}|${s.kind}:${s.metric}:${s.text}`;
                    return (
                      <Statement
                        key={`${s.metric}:${i}`}
                        s={s}
                        players={players}
                        onTrace={onTrace}
                        pinned={pinned === key}
                        onPin={() => pin(key, s.players)}
                      />
                    );
                  })}
                </ul>
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
});
