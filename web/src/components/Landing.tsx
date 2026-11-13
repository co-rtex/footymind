import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { MatchSummary, PossessionSummary, Team } from "../api/types";
import { fmtClock, matchClockSeconds, teamLabel, titleCase } from "../lib/format";
import { routeHash, useAsync } from "../lib/hooks";
import { groupByProvider, splitCommonName } from "../lib/providers";
import { Segmented } from "./Segmented";
import { StatusBlock } from "./StatusBlock";

interface Props {
  matches: MatchSummary[];
  match: MatchSummary | null;
  onSelectMatch: (id: string) => void;
  onOpenPossession: (pid: string) => void;
}

type TeamFilter = "all" | Team;

/**
 * Curated, verified examples (docs/RESEARCH.md, "Featured Possession Multiverse example").
 * Numbers quoted in the cards are from the provider's option model v2 and are recomputed live
 * when a link is opened.
 */
const FEATURED = {
  matchId: "metrica-sample-1",
  href: routeHash({
    name: "lab",
    possessionId: "metrica-sample-1:p0163",
    frame: 53708,
    moves: [{ player_id: "home_9", x: 50.7, y: -8.75 }],
  }),
};

const FEATURED_BROADCAST = {
  href: routeHash({
    name: "lab",
    possessionId: "skillcorner-1899585:p0152",
    frame: 39177,
    moves: [{ player_id: "home_25", x: 22.46, y: -5.77 }],
  }),
};

export function Landing({ matches, match, onSelectMatch, onOpenPossession }: Props) {
  return (
    <div className="landing">
      <section className="hero" aria-labelledby="hero-title">
        <p className="eyebrow">Tracking analytics lab</p>
        <h1 id="hero-title">Quantify the invisible game.</h1>
        <p className="lede">
          Space, passing options and defensive influence from real tracking data. Replay a possession, pause to see what
          the spatial model estimates, then move a player to test a constrained what-if.
        </p>
        <ol className="steps" aria-label="How to start">
          <li>
            <span className="step-n">1</span>Choose a match
          </li>
          <li>
            <span className="step-n">2</span>Open a possession
          </li>
          <li>
            <span className="step-n">3</span>Pause to analyse, then edit a scenario
          </li>
        </ol>
      </section>

      <section className="panel featured" aria-labelledby="featured-title">
        <div className="featured-body">
          <p className="eyebrow">Featured scenario · Game 1 · P1 35:48</p>
          <h2 id="featured-title" className="h-section">
            One pressing position, one risky pass
          </h2>
          <p className="muted">
            Away build up from deep under a high press. Home #9&apos;s position suppresses by far the most passing value
            here (Defensive Shadow 0.27; no other defender above 0.03). Moving him 3.5 m, within the region he could
            have reached, raises the option model&apos;s estimate for the pass that was actually played (#17 → #18) from
            about 0.45 to 0.76. All 24 ensemble members agree on the direction.
          </p>
          <p className="small muted">
            A model-based what-if under stated assumptions, not a claim about what would have happened in the match.
          </p>
        </div>
        <a className="btn btn--primary featured-cta" href={FEATURED.href} data-testid="featured-link">
          Open this scenario
        </a>
      </section>

      <section className="panel featured" aria-labelledby="featured-broadcast-title">
        <div className="featured-body">
          <p className="eyebrow">Featured scenario · SkillCorner broadcast tracking · A-League 2024/25 · P2</p>
          <h2 id="featured-broadcast-title" className="h-section">
            The same question on broadcast data
          </h2>
          <p className="muted">
            Away #27 passes to #14 (completed). Home #25&apos;s position is the most suppressive here (Defensive Shadow
            0.20). Moved within his reachable region, the option model&apos;s estimate for that pass rises from about 0.60
            to 0.80, with every ensemble member agreeing. Passer, receiver and #25 are all camera-detected; 9 of 22
            positions, away from the ball, are provider extrapolations and are drawn as estimates.
          </p>
          <p className="small muted">
            Broadcast tracking has its own option model (trained only on SkillCorner matches). Model-based what-if, not a
            claim about the match.
          </p>
        </div>
        <a className="btn btn--primary featured-cta" href={FEATURED_BROADCAST.href} data-testid="featured-broadcast-link">
          Open this scenario
        </a>
      </section>

      <section className="panel" aria-labelledby="match-title">
        <div className="panel-head">
          <h2 id="match-title" className="h-section">
            Match
          </h2>
        </div>
        <div className="match-groups" role="radiogroup" aria-label="Match" data-testid="match-groups">
          {groupByProvider(matches).map((g) => {
            const compact = g.matches.length > 4;
            const split = compact ? splitCommonName(g.matches.map((m) => m.source.name)) : null;
            return (
            <div key={g.info.id} className="match-group" role="group" aria-labelledby={`grp-${g.info.id}`}>
              <div className="match-group-head">
                <h3 id={`grp-${g.info.id}`} className="h-sub">
                  {g.info.label} <span className="match-group-count">{g.matches.length}</span>
                </h3>
                {split ? <p className="match-group-series">{split.prefix}</p> : null}
                <p className="small muted" data-testid={`provider-note-${g.info.id}`}>
                  {g.info.kind} Source:{" "}
                  <a href={g.url} target="_blank" rel="noreferrer">
                    {g.url.replace(/^https?:\/\//, "")}
                  </a>
                  .
                </p>
              </div>
              <div className={`match-cards${compact ? " match-cards--compact" : ""}`}>
                {g.matches.map((m, i) => {
                  const selected = m.match_id === match?.match_id;
                  return (
                    <button
                      key={m.match_id}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      className={`match-card${selected ? " is-selected" : ""}`}
                      onClick={() => onSelectMatch(m.match_id)}
                      title={m.source.name}
                      aria-label={`${m.source.name}, ${m.fps} Hz, ${m.n_possessions} possessions`}
                    >
                      <span className="match-card-name">{split?.labels[i] ?? m.source.name}</span>
                      <span className="match-card-meta">
                        {m.fps} Hz · {m.n_possessions} possessions · {m.n_events} events
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
            );
          })}
        </div>
      </section>

      {match ? <PossessionList key={match.match_id} match={match} onOpen={onOpenPossession} /> : null}
    </div>
  );
}

function PossessionList({ match, onOpen }: { match: MatchSummary; onOpen: (pid: string) => void }) {
  const [team, setTeam] = useState<TeamFilter>("all");
  const [shotsOnly, setShotsOnly] = useState(false);
  const poss = useAsync<PossessionSummary[]>(`poss:${match.match_id}`, (signal) =>
    api.possessions(match.match_id, 3, signal),
  );

  const rows = useMemo(
    () =>
      (poss.data ?? []).filter((p) => (team === "all" || p.team === team) && (!shotsOnly || p.has_shot)),
    [poss.data, team, shotsOnly],
  );

  return (
    <section className="panel" aria-labelledby="poss-title">
      <div className="panel-head panel-head--wrap">
        <div>
          <h2 id="poss-title" className="h-section">
            Possessions
          </h2>
          <p className="muted small">
            Open-play sequences with at least three passes.{" "}
            {poss.data ? (
              <span aria-live="polite">
                Showing {rows.length} of {poss.data.length}.
              </span>
            ) : null}
          </p>
        </div>
        <div className="filters">
          <Segmented<TeamFilter>
            label="Team"
            value={team}
            onChange={setTeam}
            options={[
              { value: "all", label: "All" },
              { value: "home", label: "Home" },
              { value: "away", label: "Away" },
            ]}
          />
          <label className="check">
            <input type="checkbox" checked={shotsOnly} onChange={(e) => setShotsOnly(e.target.checked)} />
            <span>Ends in a shot</span>
          </label>
        </div>
      </div>

      {poss.status === "error" ? (
        <StatusBlock kind="error" title="Could not load possessions" detail={poss.error.message} onRetry={poss.reload} />
      ) : !poss.data ? (
        <StatusBlock kind="loading" title="Loading possessions" />
      ) : rows.length === 0 ? (
        <StatusBlock kind="empty" title="No possessions match these filters" />
      ) : (
        <div className="table-wrap">
          <table className="table possession-table" data-testid="possession-table">
            <caption className="sr-only">Possessions in {match.source.name}</caption>
            <thead>
              <tr>
                <th scope="col">Team</th>
                <th scope="col" className="col-period">
                  Period
                </th>
                <th scope="col" className="num">
                  Clock
                </th>
                <th scope="col" className="num">
                  Duration
                </th>
                <th scope="col" className="num">
                  Passes
                </th>
                <th scope="col" className="col-seq">
                  Sequence
                </th>
                <th scope="col">
                  <span className="sr-only">Outcome</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <PossessionRow key={p.possession_id} p={p} match={match} onOpen={onOpen} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function PossessionRow({ p, match, onOpen }: { p: PossessionSummary; match: MatchSummary; onOpen: (pid: string) => void }) {
  const clock = fmtClock(matchClockSeconds(p.start_time_s, p.period, match.periods, match.fps));
  const first = titleCase(p.first_type);
  const last = [titleCase(p.last_type), p.last_subtype ? `(${titleCase(p.last_subtype)})` : ""].filter(Boolean).join(" ");
  const label = `${teamLabel(p.team)} possession, period ${p.period}, ${clock}, ${p.n_passes} ${p.n_passes === 1 ? "pass" : "passes"}${p.has_goal ? ", goal" : p.has_shot ? ", shot" : ""}`;
  return (
    <tr className="row-link" onClick={() => onOpen(p.possession_id)}>
      <td>
        <a
          className="team-chip-link"
          href={routeHash({ name: "lab", possessionId: p.possession_id })}
          aria-label={`Open ${label}`}
          onClick={(e) => e.stopPropagation()}
        >
          <span className={`team-chip team-chip--${p.team}`}>{teamLabel(p.team)}</span>
        </a>
      </td>
      <td className="col-period">P{p.period}</td>
      <td className="num">
        <span className="only-narrow">P{p.period} </span>
        {clock}
      </td>
      <td className="num">{p.duration_s.toFixed(1)} s</td>
      <td className="num">{p.n_passes}</td>
      <td className="col-seq">
        <span className="seq">
          {first || "Start"} <span aria-hidden="true">→</span>
          <span className="sr-only">to</span> {last || "End"}
        </span>
      </td>
      <td className="badges">
        {p.has_goal ? <span className="badge badge--goal">Goal</span> : null}
        {p.has_shot && !p.has_goal ? <span className="badge badge--shot">Shot</span> : null}
      </td>
    </tr>
  );
}
