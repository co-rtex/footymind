import { useState } from "react";
import type { OptionDeltaOut, PlayerOut } from "../api/types";
import { fmtSigned, playerLabel } from "../lib/format";
import { isNegligible } from "../lib/agreement";
import { comparePolicies, policyLabel, sensitivityVerdict, verdictText } from "../lib/policies";
import { ProvBadge } from "./ProvBadge";

interface Props {
  policy: string;
  staticDeltas: OptionDeltaOut[];
  policyDeltas: OptionDeltaOut[];
  players: Map<string, PlayerOut>;
  /** defenders the policy moved; 0 means the two results are the same edit */
  nSimulated: number;
}

const COMPACT_ROWS = 5;

function deltaCls(d: number | null): string {
  return isNegligible(d) ? "muted" : (d ?? 0) > 0 ? "pos" : "neg";
}

/**
 * Side-by-side pass-option changes for the same edit under static defenders ("none") and
 * under the chosen response policy, so users see how much a conclusion rests on the opponent
 * assumption. Both columns are model estimates; neither is what would have happened.
 */
export function PolicyComparison({ policy, staticDeltas, policyDeltas, players, nSimulated }: Props) {
  const [all, setAll] = useState(false);
  if (nSimulated === 0) {
    return (
      <section className="policy-compare" aria-labelledby="pc-title" data-testid="policy-comparison" data-verdict="same">
        <h3 id="pc-title" className="h-sub" tabIndex={-1}>
          Does the defender response matter?
        </h3>
        <p className="policy-verdict" data-testid="policy-verdict">
          Not for this edit: the {policyLabel(policy)} response moved no defender, so the result equals the static one.
        </p>
      </section>
    );
  }
  const c = comparePolicies(staticDeltas, policyDeltas);
  const v = sensitivityVerdict(c);
  const rows = all ? c.rows : c.rows.slice(0, COMPACT_ROWS);
  const name = policyLabel(policy);
  return (
    <section className="policy-compare" aria-labelledby="pc-title" data-testid="policy-comparison" data-verdict={v}>
      <div className="subhead">
        <h3 id="pc-title" className="h-sub" tabIndex={-1}>
          Does the defender response matter?
        </h3>
        <ProvBadge prov="estimated">model estimate</ProvBadge>
      </div>
      <p className={`policy-verdict policy-verdict--${v}`} data-testid="policy-verdict">
        {verdictText(c, policy, players)}
      </p>
      {c.rows.length > 0 ? (
        <p className="small muted">
          Change in estimated pass success for the same edit. <strong>Static</strong>: defenders stay where they were
          tracked. <strong>Response</strong>: {name}.
        </p>
      ) : null}
      {c.rows.length > 0 ? (
        <div className="table-wrap">
          <table className="table table--dense compare-table">
            <caption className="sr-only">
              Change in estimated pass success for each receiver with static defenders and with the {name} response, and
              the difference between the two
            </caption>
            <thead>
              <tr>
                <th scope="col">Receiver</th>
                <th scope="col" className="num" title="Change with defenders left where they were tracked">
                  Static Δ
                </th>
                <th scope="col" className="num" title={`Change with the ${name} response`}>
                  Response Δ
                </th>
                <th scope="col" className="num" title="Response Δ minus static Δ">
                  Gap
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.receiver_id} className={r.signFlip ? "is-flip" : undefined}>
                  <th scope="row">
                    {playerLabel(r.receiver_id, players)}
                    {r.signFlip ? <span className="pill pill--bad flip-pill">direction flips</span> : null}
                  </th>
                  <td className={`num ${deltaCls(r.staticDelta)}`}>{fmtSigned(r.staticDelta)}</td>
                  <td className={`num ${deltaCls(r.policyDelta)}`}>{fmtSigned(r.policyDelta)}</td>
                  <td className="num muted">{fmtSigned(r.gap)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {c.rows.length > COMPACT_ROWS ? (
        <button type="button" className="btn btn--ghost btn--xs" onClick={() => setAll((a) => !a)} aria-expanded={all}>
          {all ? "Show the largest changes only" : `Show all ${c.rows.length} options`}
        </button>
      ) : null}
    </section>
  );
}
