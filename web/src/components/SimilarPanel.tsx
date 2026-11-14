import { memo, useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { MatchSummary, PlayerOut, SimilarMomentsOut } from "../api/types";
import { playerLabel } from "../lib/format";
import { routeHash } from "../lib/hooks";
import {
  ballZone,
  idLabel,
  isUnvalidated,
  isValidated,
  loadMatchIndex,
  neighbourClock,
  neighbourMatchLabel,
  refHref,
  similarEmptyState,
  similarHeadline,
  type SimilarEmpty,
} from "../lib/similar";
import { matchTeams } from "../lib/teams";
import { ProvBadge } from "./ProvBadge";
import { StatusBlock } from "./StatusBlock";

interface Props {
  matchId: string;
  /** the paused replay frame (null while playing) */
  frame: number | null;
  provider: string;
  /** this match's players, labelled with team names */
  players: Map<string, PlayerOut>;
}

type Result =
  | { status: "loading"; frame: number }
  | { status: "success"; frame: number; data: SimilarMomentsOut }
  | { status: "empty"; frame: number; empty: SimilarEmpty }
  | { status: "error"; frame: number; error: Error };

const K = 8;

function EmptyNote({ empty, provider }: { empty: SimilarEmpty; provider: string }) {
  return (
    <p className="small muted similar-empty" role="status" data-testid={`similar-empty-${empty.kind}`}>
      {empty.kind === "no-decision"
        ? "No pass starts within 2 seconds of this frame, so there is no decision to compare. Step to a moment just before a pass."
        : `Similar moments are not available for ${provider} data yet: the comparison index has not been built on this server.`}
    </p>
  );
}

function Neighbours({ data, index }: { data: SimilarMomentsOut; index: Map<string, MatchSummary> }) {
  return (
    <ol className="similar-list" data-testid="similar-list">
      {data.neighbours.map((n) => {
        const summary = index.get(n.match_id);
        const where = neighbourMatchLabel(n.match_id, summary);
        const clock = neighbourClock(n, summary);
        const context = [
          clock,
          ballZone(n.ball_x_att),
          n.press_nearest_def_m !== null ? `nearest defender ${n.press_nearest_def_m.toFixed(1)} m` : null,
          n.passer_control_s !== null ? `on the ball ${n.passer_control_s.toFixed(1)} s` : null,
        ].filter(Boolean);
        return (
          <li key={`${n.match_id}@${n.frame}`} className="similar-item" data-testid="similar-item">
            <span className="similar-who">
              <strong>{idLabel(n.passer_id, matchTeams(summary))}</strong> <span className="muted">· {where}</span>
            </span>
            <span className="similar-outcome">
              <span className={`pill ${n.completed ? "pill--ok" : "pill--bad"}`}>{n.completed ? "completed" : "not completed"}</span>
              <ProvBadge prov="observed" />
            </span>
            <span className="small muted similar-context" data-testid="similar-context">
              {context.length > 0 ? context.join(" · ") : "\u00a0"}
            </span>
            <span className="similar-actions">
              <span className="small muted tabular" title="Standardised feature distance: smaller is more similar">
                distance {n.distance.toFixed(2)}
              </span>
              <a
                className="btn btn--ghost btn--xs"
                href={routeHash({ name: "at", matchId: n.match_id, frame: n.frame })}
                aria-label={`Open ${where}${clock ? ` at ${clock}` : ` at frame ${n.frame}`}`}
              >
                Open
              </a>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function ValidationNote({ data }: { data: SimilarMomentsOut }) {
  if (!isValidated(data)) return null;
  const ref = data.validation?.ref ?? null;
  const href = refHref(ref);
  return (
    <p className="small muted similar-validation" data-testid="similar-validated">
      Validated on held-out matches: the completion rate among the {data.estimate?.k ?? 50} most similar decisions
      (shrunk toward the overall rate) predicted pass completion better than the overall completion rate.
      {data.estimate ? (
        <>
          {" "}
          For this decision that estimate is <strong>{Math.round(data.estimate.completion * 100)}%</strong> (overall{" "}
          {Math.round(data.estimate.base_rate * 100)}%). The {data.neighbours.length} listed are the closest of them, shown
          for inspection; their own share is not the validated estimate.
        </>
      ) : null}
      {ref ? (
        <>
          {" "}
          Results:{" "}
          {href ? (
            <a href={href} target="_blank" rel="noreferrer">
              <code>{ref}</code>
            </a>
          ) : (
            <code title="Results file in the FootyMind repository">{ref}</code>
          )}
          .
        </>
      ) : null}
    </p>
  );
}

/**
 * "Similar moments": past pass decisions from other matches of the same provider whose
 * decision-time situation was closest, with their observed outcomes. Fetched on demand only
 * (never per scrubbed frame), and kept per frame for the session.
 */
export const SimilarPanel = memo(function SimilarPanel({ matchId, frame, provider, players }: Props) {
  const [res, setRes] = useState<Result | null>(null);
  const [index, setIndex] = useState<Map<string, MatchSummary>>(() => new Map());
  const wantIndex = res?.status === "success";
  useEffect(() => {
    if (!wantIndex) return;
    let live = true;
    void loadMatchIndex().then((m) => live && setIndex(m));
    return () => {
      live = false;
    };
  }, [wantIndex]);
  const cache = useRef(new Map<number, Result>());
  const ac = useRef<AbortController | null>(null);
  useEffect(() => () => ac.current?.abort(), []);

  // Show a cached result when returning to a frame already looked up.
  useEffect(() => {
    if (frame === null) return;
    const hit = cache.current.get(frame);
    if (hit) setRes(hit);
  }, [frame]);

  const find = useCallback(
    (f: number) => {
      ac.current?.abort();
      const ctl = new AbortController();
      ac.current = ctl;
      setRes({ status: "loading", frame: f });
      api.similar(matchId, f, K, ctl.signal).then(
        (data) => {
          const r: Result = { status: "success", frame: f, data };
          cache.current.set(f, r);
          if (!ctl.signal.aborted) setRes(r);
        },
        (e: unknown) => {
          if (ctl.signal.aborted) return;
          const empty = similarEmptyState(e);
          const r: Result = empty
            ? { status: "empty", frame: f, empty }
            : { status: "error", frame: f, error: e instanceof Error ? e : new Error(String(e)) };
          if (empty) cache.current.set(f, r);
          setRes(r);
        },
      );
    },
    [matchId],
  );

  const current = res !== null && frame !== null && res.frame === frame;
  return (
    <section className="card card--similar" aria-labelledby="similar-title" data-testid="similar-panel" data-status={res?.status ?? "idle"}>
      <div className="card-head">
        <h2 id="similar-title" className="h-card">
          Similar moments
        </h2>
        {res?.status === "success" && isUnvalidated(res.data) ? (
          <span
            className="badge badge--experimental"
            data-testid="similar-unvalidated"
            title="The held-out validation has not shown that these neighbours predict outcomes better than the overall completion rate"
          >
            unvalidated
          </span>
        ) : res?.status === "success" && isValidated(res.data) ? (
          <span className="pill pill--ok similar-validated-pill" data-testid="similar-validated-pill">
            validated held-out
          </span>
        ) : null}
      </div>

      {res === null || (!current && res.status !== "loading") ? (
        <div className="similar-start">
          {res !== null ? (
            <p className="small muted" data-testid="similar-other-frame">
              Showing results for frame {res.frame}.
            </p>
          ) : (
            <p className="small muted">
              Find passes from other matches that started in the most similar situation to the pass made at or just after
              this frame, and see how they turned out.
            </p>
          )}
          <button
            type="button"
            className="btn btn--sm"
            onClick={() => frame !== null && find(frame)}
            disabled={frame === null}
            data-testid="similar-find"
            title={frame === null ? "Pause to compare a frame" : undefined}
          >
            {res === null ? "Find similar moments" : "Find for this frame"}
          </button>
        </div>
      ) : null}

      {res?.status === "loading" ? <StatusBlock kind="loading" title="Searching other matches" compact /> : null}
      {res?.status === "empty" ? <EmptyNote empty={res.empty} provider={provider} /> : null}
      {res?.status === "error" ? (
        <StatusBlock kind="error" title="Could not find similar moments" error={res.error} onRetry={() => find(res.frame)} compact />
      ) : null}
      {res?.status === "success" ? (
        <div className="similar-body" data-testid="similar-result">
          <p className="similar-headline" data-testid="similar-headline">
            {similarHeadline(res.data)} <ProvBadge prov="observed">observed outcomes</ProvBadge>
          </p>
          <p className="small muted">
            Matched decision: pass by {playerLabel(res.data.passer_id, players)} at frame {res.data.decision_frame}. Similarity uses{" "}
            {res.data.features.length || "several"} decision-time features (positions, pressure, recent movement).{" "}
            <ProvBadge prov="derived">similarity</ProvBadge>
          </p>
          {res.data.neighbours.length > 0 ? (
            <Neighbours data={res.data} index={index} />
          ) : (
            <p className="small muted">No comparable passes were found in other matches.</p>
          )}
          <ValidationNote data={res.data} />
          <p className="disclaimer" data-testid="similar-disclaimer">
            {res.data.disclaimer}
          </p>
        </div>
      ) : null}
    </section>
  );
});
