import type { FrameAnalysisOut, PlayerOut } from "../api/types";
import { fmtMetres, playerLabel, teamLabel } from "../lib/format";
import { optionModelLabel, PHYSICS_LABEL } from "../lib/models";
import { PassOptionsTable } from "./PassOptionsTable";
import { ProvBadge } from "./ProvBadge";
import { StatusBlock } from "./StatusBlock";
import { Notices } from "./Notices";

interface Props {
  playing: boolean;
  status: "idle" | "loading" | "success" | "error";
  analysis: FrameAnalysisOut | null;
  error: Error | null;
  onRetry: () => void;
  players: Map<string, PlayerOut>;
  highlight: string | null;
  onHighlight: (id: string | null) => void;
  clock: string;
}

export const PASS_TABLE_ID = "pass-options-table";

export function AnalysisPanel({ playing, status, analysis, error, onRetry, players, highlight, onHighlight, clock }: Props) {
  if (playing) {
    return (
      <section className="card" aria-labelledby="an-title">
        <h2 id="an-title" className="h-card">
          Frame analysis
        </h2>
        <p className="muted">
          Pause (<kbd>Space</kbd>) on any frame to see estimated pitch control, ranked pass options and pressure on the
          ball carrier.
        </p>
      </section>
    );
  }
  if (status === "error" && error) {
    return (
      <section className="card" aria-labelledby="an-title">
        <h2 id="an-title" className="h-card">
          Frame analysis
        </h2>
        <StatusBlock kind="error" title="Analysis failed" detail={error.message} onRetry={onRetry} compact />
      </section>
    );
  }
  if (!analysis) {
    return (
      <section className="card" aria-labelledby="an-title" aria-busy="true">
        <h2 id="an-title" className="h-card">
          Frame analysis
        </h2>
        <StatusBlock kind="loading" title="Computing frame analysis" compact />
      </section>
    );
  }

  const a = analysis;
  const pr = a.pressure;
  return (
    <section className="card" aria-labelledby="an-title" aria-busy={status === "loading"} data-testid="analysis-panel" data-frame={a.frame}>
      <div className="card-head">
        <h2 id="an-title" className="h-card">
          Frame analysis
        </h2>
        <span className="muted small tabular">
          {clock} · frame {a.frame}
          {status === "loading" ? <span className="spinner spinner--inline" aria-label="Updating" /> : null}
        </span>
      </div>
      <p className="possession-line">
        {a.attacking_team ? (
          <>
            <span className={`team-chip team-chip--${a.attacking_team}`}>{teamLabel(a.attacking_team)}</span> in
            possession
            {a.carrier_id ? (
              <>
                {" "}
                · carrier <strong>{playerLabel(a.carrier_id, players)}</strong>
              </>
            ) : (
              <span className="muted"> · ball not at a player&apos;s feet</span>
            )}
          </>
        ) : (
          <span className="muted">No team in possession at this frame, so there is no attacking-team analysis.</span>
        )}
      </p>

      <dl className="stats">
        <div className="stat">
          <dt>
            Control area <ProvBadge prov="estimated">est.</ProvBadge>
          </dt>
          <dd>{a.control_area_m2 !== null ? `${Math.round(a.control_area_m2).toLocaleString()} m²` : "n/a"}</dd>
        </div>
        <div className="stat">
          <dt>
            In final third <ProvBadge prov="estimated">est.</ProvBadge>
          </dt>
          <dd>
            {a.control_area_final_third_m2 !== null
              ? `${Math.round(a.control_area_final_third_m2).toLocaleString()} m²`
              : "n/a"}
          </dd>
        </div>
        <div className="stat">
          <dt>
            Nearest defender <ProvBadge prov="derived" />
          </dt>
          <dd>{pr ? fmtMetres(pr.nearest_defender_distance_m) : "n/a"}</dd>
        </div>
      </dl>

      {pr ? (
        <p className="small muted pressure-line" data-testid="pressure">
          Pressure on the carrier: {playerLabel(pr.nearest_defender_id, players)} closest
          {pr.closing_speed_mps !== null ? `, closing at ${pr.closing_speed_mps.toFixed(1)} m/s` : ""};{" "}
          {pr.defenders_within_5m} defender{pr.defenders_within_5m === 1 ? "" : "s"} within 5 m,{" "}
          {pr.defenders_within_10m} within 10 m.
        </p>
      ) : null}

      <div className="subhead">
        <h3 className="h-sub">Pass options</h3>
        <ProvBadge prov="estimated">model estimate</ProvBadge>
      </div>
      <p className="model-key small" data-testid="model-key">
        <span>
          <strong>Model</strong> {optionModelLabel(a.option_model)}
          {a.option_model ? "" : " is unavailable at this frame"}
        </span>
        <span className="muted">
          <strong>Phys.</strong> {PHYSICS_LABEL}: lane clear × control at target
        </span>
      </p>
      <PassOptionsTable
        id={PASS_TABLE_ID}
        options={a.pass_options}
        players={players}
        highlight={highlight}
        onHighlight={onHighlight}
        modelLabel={optionModelLabel(a.option_model)}
      />

      <Notices warnings={a.warnings} />

      {a.option_model ? (
        <details className="model-note" data-testid="option-model-info">
          <summary>
            {optionModelLabel(a.option_model)} · {a.option_model.name}
          </summary>
          <p className="small muted">
            {a.option_model.reference}. Status: {a.option_model.status}.
          </p>
        </details>
      ) : null}
      <details className="model-note">
        <summary>
          Pitch control and physics heuristic: {a.model.name} · {a.model.status}
        </summary>
        <p className="small muted">
          {a.model.reference}. Pitch control is the probability that the attacking team would control a ball arriving at
          each point. The physics heuristic multiplies an estimated lane-clear probability by control at the target. Both
          are uncalibrated estimates, not observed outcomes.
        </p>
        <ul className="params">
          {Object.entries(a.model.params).map(([k, v]) => (
            <li key={k}>
              <code>{k}</code> {v}
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
