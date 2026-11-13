import type { OptionDeltaOut, PlayerOut } from "../api/types";
import { agreementLabel, agreementSentence, isNegligible } from "../lib/agreement";
import { fmtProb, fmtSigned, playerLabel } from "../lib/format";

function IntervalBars({ d }: { d: OptionDeltaOut }) {
  const W = 104;
  const pad = 4;
  const x = (p: number) => pad + Math.max(0, Math.min(1, p)) * (W - 2 * pad);
  const row = (y: number, lo: number | null, hi: number | null, p: number | null, cls: string) => (
    <g className={cls}>
      {lo !== null && hi !== null ? <rect x={x(lo)} y={y - 2.5} width={Math.max(1.5, x(hi) - x(lo))} height={5} rx={2.5} /> : null}
      {p !== null ? <circle cx={x(p)} cy={y} r={3.2} /> : null}
    </g>
  );
  const desc = `Observed ${fmtProb(d.observed_p)} (10 to 90 percent range ${fmtProb(d.observed_p10)} to ${fmtProb(
    d.observed_p90,
  )}); hypothetical ${fmtProb(d.hypothetical_p)} (${fmtProb(d.hypothetical_p10)} to ${fmtProb(d.hypothetical_p90)})`;
  return (
    <svg className="ibars" viewBox={`0 0 ${W} 24`} width={W} height={24} role="img" aria-label={desc}>
      <line x1={pad} x2={W - pad} y1={12} y2={12} className="ibars-axis" />
      <line x1={x(0.5)} x2={x(0.5)} y1={3} y2={21} className="ibars-mid" />
      {row(7, d.observed_p10, d.observed_p90, d.observed_p, "ibars-obs")}
      {row(17, d.hypothetical_p10, d.hypothetical_p90, d.hypothetical_p, "ibars-hyp")}
    </svg>
  );
}

interface Props {
  deltas: OptionDeltaOut[];
  players: Map<string, PlayerOut>;
  ensembleSize: number;
  highlight: string | null;
  onHighlight: (id: string | null) => void;
}

export function OptionDeltaTable({ deltas, players, ensembleSize, highlight, onHighlight }: Props) {
  const rows = [...deltas].sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0));
  if (rows.length === 0) return <p className="muted small">No pass options to compare at this frame.</p>;
  return (
    <div className="table-wrap">
      <table className="table table--dense delta-table" data-testid="option-deltas">
        <caption className="sr-only">
          Estimated pass success, observed versus hypothetical, with 10 to 90 percent ranges over {ensembleSize} parameter
          settings
        </caption>
        <thead>
          <tr>
            <th scope="col">Receiver</th>
            <th scope="col" className="num">
              Obs.
            </th>
            <th scope="col" className="num">
              Hyp.
            </th>
            <th scope="col" className="num">
              Δ
            </th>
            <th scope="col" title="10th to 90th percentile over the parameter ensemble. Grey: observed. Violet: hypothetical.">
              10–90% range
            </th>
            <th scope="col" title="Share of parameter settings whose change has the same sign">
              Agreement
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => {
            const ag = agreementLabel(d.delta_sign_agreement, d.delta);
            const negligible = isNegligible(d.delta);
            const dcls = isNegligible(d.delta) ? "muted" : (d.delta ?? 0) > 0 ? "pos" : "neg";
            return (
              <tr
                key={d.receiver_id}
                className={highlight === d.receiver_id ? "is-hl" : undefined}
                onMouseEnter={() => onHighlight(d.receiver_id)}
                onMouseLeave={() => onHighlight(null)}
              >
                <th scope="row" className="cell-player">
                  <button
                    type="button"
                    className="linkish"
                    aria-pressed={highlight === d.receiver_id}
                    onFocus={() => onHighlight(d.receiver_id)}
                    onBlur={() => onHighlight(null)}
                    onClick={() => onHighlight(highlight === d.receiver_id ? null : d.receiver_id)}
                  >
                    {playerLabel(d.receiver_id, players)}
                  </button>
                </th>
                <td className="num">{fmtProb(d.observed_p)}</td>
                <td className="num">{fmtProb(d.hypothetical_p)}</td>
                <td className={`num strong ${dcls}`}>{fmtSigned(d.delta)}</td>
                <td>
                  <IntervalBars d={d} />
                </td>
                <td>
                  <span
                    className={`agree ${ag.cls}`}
                    title={agreementSentence(d.delta_sign_agreement, ensembleSize, d.delta)}
                  >
                    {negligible ? (
                      <span className="agree-word">{ag.word}</span>
                    ) : (
                      <>
                        {d.delta_sign_agreement !== null ? `${Math.round(d.delta_sign_agreement * 100)}%` : "n/a"}
                        <span className="agree-word"> {ag.word}</span>
                      </>
                    )}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
