import type { DefensiveShadowOut, PlayerOut } from "../api/types";
import { fmtSigned, playerLabel } from "../lib/format";
import { rankShadow, SHADOW_CAVEAT, shadowStanding, suppressedReceiver } from "../lib/shadow";
import { ProvBadge } from "./ProvBadge";
import { StatusBlock } from "./StatusBlock";

interface Props {
  playing: boolean;
  status: "idle" | "loading" | "success" | "error";
  shadow: DefensiveShadowOut | null;
  /** API detail for a 422 (no ball / no possession at this frame) */
  unavailable: string | null;
  error: Error | null;
  onRetry: () => void;
  players: Map<string, PlayerOut>;
}

const STANDING_TEXT = {
  "within-noise": "within noise",
  "less-suppressive": "less suppressive than nearby positions",
  suppressive: "",
} as const;

/**
 * Experimental Defensive Shadow panel. The ranked list is also the text
 * alternative for the halos drawn on the pitch.
 */
export function ShadowPanel({ playing, status, shadow, unavailable, error, onRetry, players }: Props) {
  return (
    <section
      className="card card--shadow"
      aria-labelledby="shadow-title"
      aria-busy={status === "loading"}
      data-testid="shadow-panel"
      data-frame={shadow?.frame}
    >
      <div className="card-head">
        <h2 id="shadow-title" className="h-card">
          Defensive shadow (experimental)
        </h2>
        <ProvBadge prov="estimated">experimental estimate</ProvBadge>
      </div>
      <p className="small shadow-caveat" data-testid="shadow-caveat">
        {SHADOW_CAVEAT}
      </p>

      {playing ? (
        <p className="muted small">Pause to compute the defensive shadow for a frame (about 0.4 s per frame).</p>
      ) : status === "error" && error ? (
        <StatusBlock kind="error" title="Defensive shadow failed" detail={error.message} onRetry={onRetry} compact />
      ) : unavailable ? (
        <p className="muted small" data-testid="shadow-unavailable">
          Not available at this frame: {unavailable}.
        </p>
      ) : !shadow ? (
        <StatusBlock kind="loading" title="Computing defensive shadow (about 0.4 s)" compact />
      ) : (
        <>
          <p className="muted small tabular">
            Frame {shadow.frame} · {shadow.defenders.length} defenders compared with positions up to{" "}
            {shadow.defenders[0]?.alt_radius_m ?? "n/a"} m away · value model {shadow.value_model.replace("_", " ")}
            {status === "loading" ? <span className="spinner spinner--inline" aria-label="Updating" /> : null}
          </p>
          {shadow.defenders.length === 0 ? (
            <p className="muted small">No defenders to compare at this frame.</p>
          ) : (
            <ol className="shadow-list" data-testid="shadow-list" aria-label="Defenders ranked by estimated passing value removed">
              {rankShadow(shadow.defenders).map((d) => {
                const standing = shadowStanding(d);
                const rec = suppressedReceiver(d);
                return (
                  <li key={d.player_id} className={`shadow-row shadow-row--${standing}`} data-testid="shadow-row">
                    <span className="shadow-swatch" aria-hidden="true" />
                    <span className="shadow-who">{playerLabel(d.player_id, players)}</span>
                    <span className="shadow-val tabular" title="shadow_sum: mean summed option value at alternative positions (3.5 m ring) minus observed (+ = this position suppresses more)">
                      {fmtSigned(d.shadow_sum)}
                    </span>
                    <span className="shadow-detail small">
                      {standing === "suppressive" ? (
                        rec ? (
                          <span title="most_suppressed_delta: how much this receiver's option value would rise, on average, if the defender stood at its alternative positions (3.5 m ring)">
                            suppresses {playerLabel(rec, players)} most ({`+${d.most_suppressed_delta.toFixed(2)}`})
                          </span>
                        ) : (
                          "no single receiver stands out"
                        )
                      ) : (
                        <span className={standing === "within-noise" ? "pill pill--noise" : "muted"}>{STANDING_TEXT[standing]}</span>
                      )}
                    </span>
                  </li>
                );
              })}
            </ol>
          )}
          <p className="small muted">
            Value is shadow_sum: the summed option-model pass value at the defender&apos;s alternative positions (3.5 m ring)
            minus the observed value. Positive means this position suppresses more.
          </p>
          <details className="model-note">
            <summary>Assumptions ({shadow.assumptions.length})</summary>
            <ul className="assumptions">
              {shadow.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
            <p className="small muted">Status: {shadow.status}</p>
          </details>
        </>
      )}
    </section>
  );
}
