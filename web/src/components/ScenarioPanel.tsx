import type {
  FeasibilityOut,
  InterventionIn,
  PlayerOut,
  PolicyInfo,
  ResponsePolicy,
  ScenarioResponse,
  SummaryDict,
  ValueModel,
} from "../api/types";
import { MAX_INTERVENTIONS } from "../api/types";
import { fmtMetres, fmtSigned, humaniseIds, playerLabel } from "../lib/format";
import { agreementSentence } from "../lib/agreement";
import { HORIZON_S } from "../lib/scenario";
import { optionModelLabel, PHYSICS_LABEL, valueModelLabel } from "../lib/models";
import { POLICY_BADGE_TEXT, policyBadge, policyFitRange, policyLabel, STATIC_POLICY } from "../lib/policies";
import { PolicyBadge } from "./PolicyBadge";
import { OptionDeltaTable } from "./OptionDeltaTable";
import { PolicyComparison } from "./PolicyComparison";
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
  /** policies the backend offers (or the fallback list on older backends) */
  policies: PolicyInfo[];
  /** the same edit with static defenders, for the sensitivity comparison (only when policy is not static) */
  staticStatus: "idle" | "loading" | "success" | "error";
  staticScenario: ScenarioResponse | null;
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

/** True when every moved player belongs to the defending team (known attacking team only). */
function onlyDefendersMoved(
  moves: InterventionIn[],
  players: Map<string, PlayerOut>,
  attacking: string | null,
): boolean {
  if (!attacking || moves.length === 0) return false;
  return moves.every((m) => {
    const t = players.get(m.player_id)?.team;
    return t !== undefined && t !== attacking;
  });
}

function FeasibilityLine({ f, players }: { f: FeasibilityOut; players: Map<string, PlayerOut> }) {
  return (
    <li className={f.supported ? "feas feas--ok" : "feas feas--bad"}>
      <span className="feas-head">
        <strong>{playerLabel(f.player_id, players)}</strong> moved {fmtMetres(f.displacement_m)}
        <span className={`pill ${f.supported ? "pill--ok" : "pill--bad"}`}>{f.supported ? "supported" : "not supported"}</span>
      </span>
      <span className="small muted" data-testid="feasibility-detail">
        {f.reachable ? "Reachable" : "Not reachable"}: {f.reachable ? "within" : "more than"} {fmtMetres(f.reachable_radius_m)}{" "}
        of the observed position {HORIZON_S} s earlier (the scenario&apos;s horizon) ·{" "}
        {f.inside_pitch ? "inside the pitch" : "outside the pitch"} · nearest other player {fmtMetres(f.min_separation_m)}
        {typeof f.kinematic_reachable === "boolean" ? (
          <span data-testid="feasibility-kinematic">
            {" "}
            · {f.kinematic_reachable ? "inside" : "outside"} the kinematic envelope (diagnostic only)
          </span>
        ) : null}
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

function PolicyPicker({
  policy,
  policies,
  info,
  onPolicy,
  beyondFit,
  players,
}: {
  policy: ResponsePolicy;
  policies: PolicyInfo[];
  info: PolicyInfo | null;
  onPolicy: (p: ResponsePolicy) => void;
  /** moved players whose displacement exceeds the policy's fitted range (current result only) */
  beyondFit: FeasibilityOut[];
  players: Map<string, PlayerOut>;
}) {
  const id = "policy-select";
  const options: PolicyInfo[] = policies.some((q) => q.name === policy)
    ? policies
    : [...policies, { name: policy, description: "", assumptions: [], validated: false, validation_ref: null, status: null }];
  const badge = policyBadge(info);
  const fit = policy !== STATIC_POLICY ? policyFitRange(info) : null;
  return (
    <div className="policy-picker" data-testid="policy-picker">
      <div className="policy-picker-row">
        <label htmlFor={id} className="segmented-label">
          Defender response
        </label>
        <select
          id={id}
          className="select"
          value={policy}
          onChange={(e) => onPolicy(e.target.value)}
          data-testid="policy-select"
          aria-describedby={`${id}-desc`}
        >
          {options.map((q) => {
            const b = policyBadge(q);
            return (
              <option key={q.name} value={q.name} data-badge={b ?? ""}>
                {policyLabel(q.name)}
                {b ? ` (${b})` : ""}
              </option>
            );
          })}
        </select>
        <PolicyBadge info={info} testId="policy-badge" />
      </div>
      <div id={`${id}-desc`} className="small muted policy-note" data-testid="policy-description">
        <p>
          {info?.description ||
            (policy === STATIC_POLICY ? "Only the players you move change position." : "No description available.")}
        </p>
        {badge ? (
          <p data-testid="policy-verdict-note">
            <strong>{badge}:</strong> {POLICY_BADGE_TEXT[badge]}
            {info?.status ? (
              <>
                {" "}
                <span data-testid="policy-status">Backend verdict: {info.status}.</span>
              </>
            ) : null}
            {info?.validation_ref ? <> Evidence: {info.validation_ref}.</> : null}
          </p>
        ) : null}
        {fit ? (
          <p className="policy-fit" data-testid="policy-fit-range">
            <strong>Fitted range:</strong> {fit.sharePct}% of the real unexpected attacker movement it was fitted on is
            under about {fit.limitM} m in {fit.horizonS} s. Edits larger than about {fit.limitM} m go beyond that, so its
            response to them is an extrapolation.
            {beyondFit.length > 0 ? (
              <span data-testid="policy-fit-exceeded">
                {" "}
                This edit moves{" "}
                {beyondFit.map((f, i) => (
                  <span key={f.player_id}>
                    {i > 0 ? ", " : ""}
                    {playerLabel(f.player_id, players)} {fmtMetres(f.displacement_m)}
                  </span>
                ))}
                .
              </span>
            ) : null}
          </p>
        ) : null}
      </div>
    </div>
  );
}

export function ScenarioPanel(p: Props) {
  const { scenario, players } = p;
  const top = scenario
    ? [...scenario.option_deltas].sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0))[0]
    : undefined;
  // The response's own description wins: it describes the policy that produced these moves.
  const policyInfo =
    (scenario?.policy && scenario.policy.name === p.policy ? scenario.policy : null) ??
    p.policies.find((q) => q.name === p.policy) ??
    null;
  // Policy-specific statements only once the result for the current choice has arrived.
  const fresh = p.status === "success";
  const simulated = fresh ? (scenario?.movements.filter((m) => m.provenance === "simulated") ?? []) : [];
  const fit = p.policy !== STATIC_POLICY ? policyFitRange(policyInfo) : null;
  const beyondFit =
    fresh && fit && scenario
      ? // compared at the precision shown (0.1 m), so "2.5 m" is never listed as beyond "about 2.5 m"
        scenario.feasibility.filter((f) => f.kind !== "simulated" && Math.round(f.displacement_m * 10) / 10 > fit.limitM)
      : [];

  return (
    <section className="card card--scenario" aria-labelledby="sc-title" data-testid="scenario-panel" data-status={p.status}>
      <div className="card-head">
        <h2 id="sc-title" className="h-card" tabIndex={-1}>
          Possession Multiverse
        </h2>
        <span className="muted small tabular">anchored at {p.frameLabel}</span>
      </div>

      {p.interventions.length === 0 ? (
        <div className="howto">
          <p>
            <strong>Drag any player</strong> to a new position. While you drag, the dashed circle shows the region the
            backend marks as reachable for that player: a top-speed limit applied from their observed position a moment
            earlier. Each moved player&apos;s reach radius is listed under Evidence &amp; assumptions.
          </p>
          <p className="small muted">
            Up to {MAX_INTERVENTIONS} players. The model is re-run on the edited frame and compared with the observed one.
            Without a mouse, or on a small screen: choose a player in the <strong>Player</strong> menu above the
            pitch and type the new position in the player card.
          </p>
          {!p.ready ? <StatusBlock kind="loading" title="Waiting for the observed analysis" compact /> : null}
        </div>
      ) : null}

      <PolicyPicker
        policy={p.policy}
        policies={p.policies}
        info={policyInfo}
        onPolicy={p.onPolicy}
        beyondFit={beyondFit}
        players={players}
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
          {simulated.map((m) => (
            <li key={`sim-${m.player_id}`} data-testid="simulated-move">
              <ProvBadge prov="simulated" />
              <span>
                {playerLabel(m.player_id, players)} moved {fmtMetres(Math.hypot(m.to_x - m.from_x, m.to_y - m.from_y))} by
                the {policyLabel(p.policy)} response
                {m.reason ? <span className="muted"> · {humaniseIds(m.reason, players)}</span> : null}
              </span>
            </li>
          ))}
          {fresh && scenario && p.policy !== STATIC_POLICY && simulated.length === 0 ? (
            <li className="small muted moves-note" data-testid="no-simulated-moves">
              The {policyLabel(p.policy)} response moved no defender for this edit.
              {onlyDefendersMoved(p.interventions, players, scenario.observed.attacking_team) ? (
                <span data-testid="response-hint">
                  {" "}
                  You moved only defenders; the response models react to a moved attacker or ball carrier (see the
                  description above). Move an attacker to see one respond.
                </span>
              ) : null}
            </li>
          ) : null}
        </ul>
      ) : null}

      {p.status === "loading" ? <StatusBlock kind="loading" title="Re-running the model on the edited frame" compact /> : null}
      {p.status === "error" && p.error ? (
        <StatusBlock kind="error" title="Scenario failed" error={p.error} onRetry={p.onRetry} compact />
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
          {/* scrolls on its own on very narrow screens instead of widening the panel */}
          <div className="table-wrap">
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
          </div>

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

          {!fresh ? null : p.policy !== STATIC_POLICY ? (
            p.staticScenario && p.staticScenario.value_model === scenario.value_model ? (
              <PolicyComparison
                policy={p.policy}
                staticDeltas={p.staticScenario.option_deltas}
                policyDeltas={scenario.option_deltas}
                players={players}
                nSimulated={simulated.length}
              />
            ) : p.staticStatus === "error" ? (
              <p className="small muted" data-testid="policy-comparison-error">
                The static comparison could not be computed, so the effect of the defender response is not shown.
              </p>
            ) : (
              <StatusBlock kind="loading" title="Computing the same edit with static defenders" compact />
            )
          ) : (
            <p className="small muted" data-testid="policy-static-hint">
              Defenders stay where they were tracked. Choose a defender response above to see how much this result depends
              on that assumption.
            </p>
          )}

          <section className="evidence" aria-labelledby="ev-title" data-testid="evidence">
            <h3 id="ev-title" className="h-sub" tabIndex={-1}>
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
            {scenario.policy && scenario.policy.name !== STATIC_POLICY && scenario.policy.assumptions.length > 0 ? (
              <details data-testid="policy-assumptions">
                <summary>
                  {policyLabel(scenario.policy.name)} response: assumptions ({scenario.policy.assumptions.length})
                </summary>
                <ul className="assumptions">
                  {scenario.policy.assumptions.map((a, i) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              </details>
            ) : null}
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
