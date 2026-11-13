import type { PassOptionOut, PlayerOut } from "../api/types";
import { fmtMetres, fmtProb, playerLabel } from "../lib/format";

interface Props {
  id?: string;
  options: PassOptionOut[];
  players: Map<string, PlayerOut>;
  highlight: string | null;
  onHighlight: (id: string | null) => void;
}

/** Ranked pass options. Doubles as the text alternative for the pass lines on the canvas. */
export function PassOptionsTable({ id, options, players, highlight, onHighlight }: Props) {
  const ranked = [...options].sort((a, b) => b.p_success - a.p_success);
  if (ranked.length === 0) {
    return (
      <p className="muted small" id={id} data-testid="pass-options-empty">
        No pass options at this frame.
      </p>
    );
  }
  return (
    <div className="table-wrap" id={id}>
      <table className="table table--dense options-table" data-testid="pass-options">
        <caption className="sr-only">
          Pass options from the ball, ranked by estimated probability of success (model estimate, uncalibrated). Each row
          lists the closest defender to the passing lane.
        </caption>
        <thead>
          <tr>
            <th scope="col" title="Second line: the defender closest to the passing lane and their perpendicular distance">
              Receiver
            </th>
            <th scope="col" title="Estimated P(success) = P(lane clear) x P(control at target)">
              P(success)
            </th>
            <th scope="col" className="num" title="Estimated probability that no defender intercepts in the lane">
              Lane
            </th>
            <th scope="col" className="num" title="Estimated attacking pitch control at the target">
              Ctrl
            </th>
            <th scope="col" className="num" title="Pass distance in metres">
              Dist
            </th>
            <th scope="col" className="num" title="Forward progress towards the opponent goal in metres (negative = backwards)">
              Fwd
            </th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((o) => (
            <tr
              key={o.receiver_id}
              className={highlight === o.receiver_id ? "is-hl" : undefined}
              onMouseEnter={() => onHighlight(o.receiver_id)}
              onMouseLeave={() => onHighlight(null)}
            >
              <th scope="row" className="cell-player">
                <button
                  type="button"
                  className="linkish"
                  aria-pressed={highlight === o.receiver_id}
                  onFocus={() => onHighlight(o.receiver_id)}
                  onBlur={() => onHighlight(null)}
                  onClick={() => onHighlight(highlight === o.receiver_id ? null : o.receiver_id)}
                >
                  {playerLabel(o.receiver_id, players)}
                </button>
                <span className="cell-sub">
                  {o.lane_closest_defender_id ? (
                    <>
                      <span className="sr-only">closest lane defender </span>
                      {playerLabel(o.lane_closest_defender_id, players)} · {fmtMetres(o.lane_closest_defender_perp_m)}
                    </>
                  ) : (
                    "no lane defender"
                  )}
                </span>
              </th>
              <td>
                <span className="pbar" aria-hidden="true">
                  <span className="pbar-fill" style={{ width: `${Math.round(o.p_success * 100)}%` }} />
                </span>
                <span className="pbar-val">{fmtProb(o.p_success)}</span>
              </td>
              <td className="num">{fmtProb(o.p_lane_clear)}</td>
              <td className="num">{fmtProb(o.p_control_at_target)}</td>
              <td className="num">{o.distance_m.toFixed(0)}</td>
              <td className={`num ${o.forward_progress_m >= 0.5 ? "pos" : o.forward_progress_m <= -0.5 ? "neg" : ""}`}>
                {o.forward_progress_m >= 0.5 ? "+" : o.forward_progress_m <= -0.5 ? "−" : ""}
                {Math.abs(o.forward_progress_m).toFixed(0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
