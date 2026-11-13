import type {
  FeasibilityOut,
  InterventionIn,
  PlayerOut,
  ResponsePolicy,
  ScenarioResponse,
  SummaryDict,
  ValueModel,
} from "../api/types";
import { MAX_INTERVENTIONS } from "../api/types";
import { fmtMetres, fmtSigned, playerLabel } from "../lib/format";
import { agreementSentence } from "../lib/agreement";
import { optionModelLabel, PHYSICS_LABEL, valueModelLabel } from "../lib/models";
import { OptionDeltaTable } from "./OptionDeltaTable";
import { ProvBadge } from "./ProvBadge";
import { Segmented } from "./Segmented";
import { StatusBlock } from "./StatusBlock";
import { Notices } from "./Notices";

interface Props {
  ready: boolean;
  frameLabel: string;
  interventions: InterventionIn[];
  policy: ResponsePolicy;
  onPolicy: (p: ResponsePolicy) => void;
  valueModel: ValueModel;
  onValueModel: (v: ValueModel) => void;
  status: "idle" | "loading" | "success" | "error";
  scenario: ScenarioResponse | null;
  error: Error | null;
  onRetry: () => void;
  onRemove: (playerId: string) => void;
  onReset: () => void;
  players: Map<string, PlayerOut>;
  highlight: string | null;
  onHighlight: (id: string | null) => void;
  notice: string | null;
}

const SUMMARY_ROWS: { key: string; label: string; kind: "area" | "player" | "prob" | "count" }[] = [
  { key: "control_area_m2", label: "Attacking control", kind: "area" },
  { key: "control_area_final_third_m2", label: "Control in final third", kind: "area" },
  { key: "best_option_receiver", label: "Best pass option", kind: "player" },
  { key: "best_option_p_success_model", label: "Best option (model)", kind: "prob" },
  { key: "best_option_p_success_physics", label: "Best option (physics)", kind: "prob" },
  { key: "n_options_physics_over_0_5", label: "Physics options > 0.5", kind: "count" },
  // pre-option-model key names, still shown if an older backend sends them
  { key: "best_option_p_success", label: "Best option P(success)", kind: "prob" },
  { key: "n_options_p_over_0_5", label: "Options with P > 0.5", kind: "count" },
];

function fmtSummary(v: number | string | null | undefined, kind: string, players: Map<string, PlayerOut>): string {
  if (v === null || v === undefined) return "n/a";
  if (kind === "player") return playerLabel(String(v), players);
  if (typeof v !== "number") return String(v);
  if (kind === "area") return `${Math.round(v).toLocaleString()} m²`;
  if (kind === "prob") return v.toFixed(2);
  return String(v);
}

function fmtSummaryDelta(o: SummaryDict, h: SummaryDict, key: string, kind: string): string | null {
  const a = o[key];
  const b = h[key];
  if (typeof a !== "number" || typeof b !== "number") return null;
  const d = b - a;
  if (kind === "area") return `${d >= 0 ? "+" : "−"}${Math.abs(Math.round(d)).toLocaleString()} m²`;
  if (kind === "prob") return fmtSigned(d);
  return d === 0 ? "±0" : `${d > 0 ? "+" : "−"}${Math.abs(d)}`;
}

function FeasibilityLine({ f, players }: { f: FeasibilityOut; players: Map<string, PlayerOut> }) {
  return (
    <li className={f.supported ? "feas feas--ok" : "feas feas--bad"}>
      <span className="feas-head">
        <strong>{playerLabel(f.player_id, players)}</strong> moved {fmtMetres(f.displacement_m)}
        <span className={`pill ${f.supported ? "pill--ok" : "pill--bad"}`}>{f.supported ? "supported" : "not supported"}</span>
      </span>
      <span className="small muted">
        {f.reachable ? "Reachable" : "Not reachable"} within {fmtMetres(f.reachable_radius_m)} of the observed position 1.5 s
        earlier · {f.inside_pitch ? "inside the pitch" : "outside the pitch"} · nearest other player{" "}
        {fmtMetres(f.min_separation_m)}
      </span>
      {f.notes.length > 0 ? (
        <ul className="small muted feas-notes">
          {f.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

export function ScenarioPanel(p: Props) {
  const { scenario, players } = p;
  const top = scenario
    ? [...scenario.option_deltas].sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0))[0]
    : undefined;

  return (
    <section className="card card--scenario" aria-labelledby="sc-title" data-testid="scenario-panel" data-status={p.status}>
      <div className="card-head">
        <h2 id="sc-title" className="h-card">
          Possession Multiverse
        </h2>
        <span className="muted small tabular">anchored at {p.frameLabel}</span>
      </div>

      {p.interventions.length === 0 ? (
        <div className="howto">
          <p>
            <strong>Drag any player</strong> to a new position. While you drag, the dashed circle shows where that player
            could have reached at up to 7 m/s from their observed position 1.5 s earlier.
          </p>
          <p className="small muted">
            Up to {MAX_INTERVENTIONS} players. The model is re-run on the edited frame and compared with the observed one.
            Keyboard users: select a player and use the move form in the player card.
          </p>
          {!p.ready ? <StatusBlock kind="loading" title="Waiting for the observed analysis" compact /> : null}
        </div>
      ) : null}

      <Segmented<ResponsePolicy>
        label="Opponent response"
        value={p.policy}
        onChange={p.onPolicy}
        testId="policy"
        options={[
          { value: "none", label: "No response", title: "Only the players you move change position" },
          {
            value: "marker_follows",
            label: "Marker follows",
            title: "Rule: when you move an attacker, the nearest defender within 10 m shifts by half the displacement",
          },
        ]}
      />

      <Segmented<ValueModel>
        label="Compare using"
        value={p.valueModel}
        onChange={p.onValueModel}
        testId="value-model"
        options={[
          { value: "option_model", label: "Option model", title: optionModelLabel(scenario?.observed.option_model) },
          { value: "physics", label: "Physics heuristic", title: PHYSICS_LABEL },
        ]}
      />

      {p.notice ? (
        <p className="inline-warn" role="status">
          {p.notice}
        </p>
      ) : null}

      {p.interventions.length > 0 ? (
        <ul className="moves" aria-label="Moved players">
          {p.interventions.map((iv) => (
            <li key={iv.player_id}>
              <ProvBadge prov="hypothetical" />
              <span>
                {playerLabel(iv.player_id, players)} to ({iv.x.toFixed(1)}, {iv.y.toFixed(1)}) m
              </span>
              <button
                type="button"
                className="btn btn--ghost btn--xs"
                onClick={() => p.onRemove(iv.player_id)}
                aria-label={`Undo move of ${playerLabel(iv.player_id, players)}`}
              >
                Undo
              </button>
            </li>
          ))}
          {scenario?.movements
            .filter((m) => m.provenance === "simulated")
            .map((m) => (
              <li key={`sim-${m.player_id}`}>
                <ProvBadge prov="simulated" />
                <span>
                  {playerLabel(m.player_id, players)} shifted {fmtMetres(Math.hypot(m.to_x - m.from_x, m.to_y - m.from_y))}
                  <span className="muted"> ({m.reason})</span>
                </span>
              </li>
            ))}
        </ul>
      ) : null}

      {p.status === "loading" ? <StatusBlock kind="loading" title="Re-running the model on the edited frame" compact /> : null}
      {p.status === "error" && p.error ? (
        <StatusBlock kind="error" title="Scenario failed" detail={p.error.message} onRetry={p.onRetry} compact />
      ) : null}

      {scenario && p.interventions.length > 0 ? (
        <>
          <div className={`verdict ${scenario.supported ? "verdict--ok" : "verdict--bad"}`} data-testid="scenario-verdict">
            <strong>{scenario.supported ? "Physically plausible edit" : "Edit not supported"}</strong>
            <span className="small">
              {scenario.supported
                ? "Every moved player passes the reachability, pitch and separation checks."
                : "At least one moved player fails a feasibility check. Read the deltas as illustrative only."}
            </span>
          </div>

          {top && top.delta !== null ? (
            <p className="headline" data-testid="scenario-headline">
              Largest change: pass to <strong>{playerLabel(top.receiver_id, players)}</strong>{" "}
              <span className={top.delta > 0 ? "pos" : top.delta < 0 ? "neg" : ""}>{fmtSigned(top.delta)}</span>{" "}
              {scenario.value_model === "physics" ? "physics-heuristic" : "option-model"} probability{" "}
              {`(${agreementSentence(top.delta_sign_agreement, scenario.ensemble_size, top.delta)}).`}
            </p>
          ) : null}

          <div className="subhead">
            <h3 className="h-sub">Summary</h3>
            <ProvBadge prov="estimated">model estimate</ProvBadge>
          </div>
          <table className="table table--dense summary-table">
            <thead>
              <tr>
                <th scope="col">Quantity</th>
                <th scope="col" className="num">
                  Observed
                </th>
                <th scope="col" className="num">
                  Hypothetical
                </th>
                <th scope="col" className="num">
                  Change
                </th>
              </tr>
            </thead>
            <tbody>
              {SUMMARY_ROWS.filter((r) => r.key in scenario.summary_observed || r.key in scenario.summary_hypothetical).map(
                (r) => (
                  <tr key={r.key}>
                    <th scope="row">{r.label}</th>
                    <td className="num">{fmtSummary(scenario.summary_observed[r.key], r.kind, players)}</td>
                    <td className="num">{fmtSummary(scenario.summary_hypothetical[r.key], r.kind, players)}</td>
                    <td className="num muted">{fmtSummaryDelta(scenario.summary_observed, scenario.summary_hypothetical, r.key, r.kind) ?? ""}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>

          <div className="subhead">
            <h3 className="h-sub">Pass options: observed vs hypothetical</h3>
            <ProvBadge prov="estimated">model estimate</ProvBadge>
          </div>
          <p className="small muted" data-testid="delta-model">
            Values: {valueModelLabel(scenario.value_model, scenario.observed.option_model)}.
          </p>
          <OptionDeltaTable
            deltas={scenario.option_deltas}
            players={players}
            ensembleSize={scenario.ensemble_size}
            highlight={p.highlight}
            onHighlight={p.onHighlight}
          />
          <p className="small muted">
            Ranges are the 10th to 90th percentile across {scenario.ensemble_size} perturbed model parameter settings.
            Agreement is the share of settings in which the change has the same sign.
          </p>

          <section className="evidence" aria-labelledby="ev-title" data-testid="evidence">
            <h3 id="ev-title" className="h-sub">
              Evidence &amp; assumptions
            </h3>
            <ul className="feas-list">
              {scenario.feasibility.map((f) => (
                <FeasibilityLine key={f.player_id} f={f} players={players} />
              ))}
            </ul>
            <Notices warnings={scenario.warnings} title="Scenario warnings" />
            <details>
              <summary>Assumptions ({scenario.assumptions.length})</summary>
              <ul className="assumptions">
                {scenario.assumptions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </details>
            <p className="disclaimer">{scenario.disclaimer}</p>
          </section>
        </>
      ) : null}

      <div className="card-actions">
        <button
          type="button"
          className="btn"
          onClick={p.onReset}
          disabled={p.interventions.length === 0 && !scenario}
          data-testid="reset-scenario"
        >
          Reset to observed
        </button>
      </div>
    </section>
  );
}
