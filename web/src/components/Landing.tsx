import { useMemo, useState, type MouseEvent } from "react";
import { api } from "../api/client";
import type { MatchSummary, PossessionSummary, Team } from "../api/types";
import { FEATURED, featuredHref } from "../lib/featured";
import { fmtClock, matchClockSeconds, titleCase } from "../lib/format";
import { hasTeamNames, matchFacts, matchTeams, matchTitle, teamName } from "../lib/teams";
import { routeHash, useAsync } from "../lib/hooks";
import { groupByProvider, splitCommonName } from "../lib/providers";
import { FeaturedCard } from "./FeaturedCard";
import { ProvBadge } from "./ProvBadge";
import { ProvenanceKey } from "./ProvenanceKey";
import { Segmented } from "./Segmented";
import { StatusBlock } from "./StatusBlock";

interface Props {
  matches: MatchSummary[];
  match: MatchSummary | null;
  onSelectMatch: (id: string) => void;
  onOpenPossession: (pid: string) => void;
}

type TeamFilter = "all" | Team;

/** Possessions listed before "Show all". */
export const POSSESSION_PAGE = 20;

/**
 * On a phone the possession list starts far below the match cards, so choosing a match would
 * otherwise change nothing visible. Bring the list's heading into view there (no animation).
 */
function revealPossessions(): void {
  if (typeof window === "undefined" || !window.matchMedia?.("(max-width: 720px)").matches) return;
  requestAnimationFrame(() => {
    const el = document.getElementById("poss-title");
    if (el && el.getBoundingClientRect().top > window.innerHeight * 0.6) el.scrollIntoView({ block: "start" });
  });
}

const PILLARS: { title: string; text: string; prov: "estimated" | "derived" | "hypothetical" }[] = [
  { title: "Space", text: "Which team would win a ball played to each spot on the pitch.", prov: "estimated" },
  { title: "Pressure", text: "How close the defenders are to the ball, and whether they are closing in.", prov: "derived" },
  { title: "Passing options", text: "Every teammate the ball could go to, with an estimated chance it arrives.", prov: "estimated" },
  { title: "What if?", text: "Move a player where they could have been, and compare the estimates.", prov: "hypothetical" },
];

export function Landing({ matches, match, onSelectMatch, onOpenPossession }: Props) {
  const tour = FEATURED[0]!;
  const summaries = useMemo(() => new Map(matches.map((m) => [m.match_id, m])), [matches]);
  const tourTeams = matchTeams(summaries.get(tour.matchId) ?? null);
  const steps = tour.steps(tourTeams);
  return (
    <div className="landing">
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">Football tracking analytics lab</p>
          <h1 id="hero-title">The Invisible Game</h1>
          <p className="hero-sub">
            Event data says a pass went from #17 to #18. It cannot say how much space #18 had, who was pressing, or which
            other passes were on.
          </p>
          <p className="lede">
            FootyMind reads the positions of all 22 players from real tracking data and estimates that invisible layer.
            Then it lets you ask <em>what if</em>: move a player and compare the model&apos;s estimates with what was
            observed.
          </p>
          {/* stacked layouts only: the tour card is further down there */}
          <a className="btn btn--primary btn--lg hero-cta" href={featuredHref(tour, "tour")} data-testid="tour-link-hero">
            Start the guided tour <span aria-hidden="true">→</span>
          </a>
          <dl className="pillars" aria-label="What FootyMind shows">
            {PILLARS.map((p) => (
              <div key={p.title} className={`pillar pillar--${p.prov}`}>
                <dt>
                  {p.title} <ProvBadge prov={p.prov} />
                </dt>
                <dd>{p.text}</dd>
              </div>
            ))}
          </dl>
        </div>

        <section className="tour-card" aria-labelledby="tour-title" data-testid="tour-card">
          <p className="eyebrow">New here? Start with one real moment</p>
          <h2 id="tour-title" className="h-section">
            A three-step guided tour
          </h2>
          <p className="small muted">{tour.title}</p>
          <ol className="tour-steps">
            <li>
              <span className="step-n" aria-hidden="true">
                1
              </span>
              <span>
                <strong>Watch</strong> {steps.watch}
              </span>
            </li>
            <li>
              <span className="step-n" aria-hidden="true">
                2
              </span>
              <span>
                <strong>Understand</strong> {steps.why}
              </span>
            </li>
            <li>
              <span className="step-n" aria-hidden="true">
                3
              </span>
              <span>
                <strong>Test a what-if</strong> {steps.move}
              </span>
            </li>
          </ol>
          <a className="btn btn--primary btn--lg" href={featuredHref(tour, "tour")} data-testid="tour-link">
            Start the guided tour <span aria-hidden="true">→</span>
          </a>
          <a className="small tour-skip" href="#explore" onClick={skipToExplore}>
            Or browse every match and possession
          </a>
        </section>
      </section>

      <section className="section" aria-labelledby="featured-title">
        <div className="section-head">
          <h2 id="featured-title" className="h-section">
            Featured situations
          </h2>
          <p className="small muted">
            Three real passes, each picked by a documented search: two for robust model changes, one for a change that
            depends on how the defenders are assumed to respond. The numbers are
            computed live from the tracking data when this page loads. Each what-if is a model-based comparison, not a
            claim about what would have happened in the match.
          </p>
        </div>
        <div className="feature-grid">
          {FEATURED.map((f) => (
            <FeaturedCard
              key={f.id}
              f={f}
              match={summaries.get(f.matchId) ?? null}
              testId={f.id === "press" ? "featured-link" : `featured-${f.id}-link`}
            />
          ))}
        </div>
      </section>

      <section className="section" aria-labelledby="labels-title">
        <div className="section-head">
          <h2 id="labels-title" className="h-section">
            How to read what you see
          </h2>
          <p className="small muted">
            Every value and every player on the pitch carries one of five labels, so you can always tell what was measured
            from what a model or you supplied.
          </p>
        </div>
        <ProvenanceKey headingId="labels-title" />
      </section>

      <section className="panel" aria-labelledby="match-title" id="explore">
        <div className="panel-head">
          <div>
            <h2 id="match-title" className="h-section" tabIndex={-1}>
              Explore any possession
            </h2>
            <p className="small muted">Choose a match, then open a possession to replay and analyse it.</p>
          </div>
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
                    // Real team names when the provider publishes them; anonymised matches keep the source name.
                    const named = hasTeamNames(matchTeams(m));
                    const name = named ? `${m.home_label} vs ${m.away_label}` : (split?.labels[i] ?? m.source.name);
                    const facts = matchFacts(m);
                    return (
                      <button
                        key={m.match_id}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        className={`match-card${selected ? " is-selected" : ""}`}
                        onClick={() => {
                          onSelectMatch(m.match_id);
                          revealPossessions();
                        }}
                        title={m.source.name}
                        aria-label={`${named ? matchTitle(m) : m.source.name}, ${m.fps} Hz, ${m.n_possessions} possessions`}
                        data-testid={`match-card-${m.match_id}`}
                      >
                        <span className="match-card-name">{name}</span>
                        {facts ? <span className="match-card-facts">{facts}</span> : null}
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

/** In-page link to the match browser: scroll there and move focus to its heading. */
function skipToExplore(e: MouseEvent<HTMLAnchorElement>): void {
  // keep the hash router on the current route; "#explore" is not a route
  e.preventDefault();
  const h = document.getElementById("match-title");
  h?.scrollIntoView({ block: "start" });
  h?.focus({ preventScroll: true });
}

function PossessionList({ match, onOpen }: { match: MatchSummary; onOpen: (pid: string) => void }) {
  const [team, setTeam] = useState<TeamFilter>("all");
  const [shotsOnly, setShotsOnly] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const teams = matchTeams(match);
  const poss = useAsync<PossessionSummary[]>(`poss:${match.match_id}`, (signal) =>
    api.possessions(match.match_id, 3, signal),
  );

  const rows = useMemo(
    () =>
      (poss.data ?? []).filter((p) => (team === "all" || p.team === team) && (!shotsOnly || p.has_shot)),
    [poss.data, team, shotsOnly],
  );
  const visible = showAll ? rows : rows.slice(0, POSSESSION_PAGE);

  return (
    <section className="panel" aria-labelledby="poss-title">
      <div className="panel-head panel-head--wrap">
        <div>
          <h2 id="poss-title" className="h-section">
            Possessions
          </h2>
          <p className="small" data-testid="possessions-match">
            {matchTitle(match)}
          </p>
          <p className="muted small">
            Open-play sequences with at least three passes.{" "}
            {poss.data ? (
              <span aria-live="polite" data-testid="possessions-count">
                {rows.length} of {poss.data.length} match the filters.
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
              { value: "home", label: teamName("home", teams) },
              { value: "away", label: teamName("away", teams) },
            ]}
          />
          <label className="check">
            <input type="checkbox" checked={shotsOnly} onChange={(e) => setShotsOnly(e.target.checked)} />
            <span>Ends in a shot</span>
          </label>
        </div>
      </div>

      {poss.status === "error" ? (
        <StatusBlock kind="error" title="Could not load possessions" error={poss.error} subject="the possessions of this match" onRetry={poss.reload} />
      ) : !poss.data ? (
        <StatusBlock kind="loading" title="Loading possessions" />
      ) : rows.length === 0 ? (
        <StatusBlock
          kind="empty"
          title="No possessions match these filters"
          action={
            <button
              type="button"
              className="btn btn--sm"
              onClick={() => {
                setTeam("all");
                setShotsOnly(false);
              }}
            >
              Clear filters
            </button>
          }
        />
      ) : (
        <>
          <div className="table-wrap">
            <table className="table possession-table" data-testid="possession-table">
              <caption className="sr-only">Possessions in {matchTitle(match)}</caption>
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
                {visible.map((p) => (
                  <PossessionRow key={p.possession_id} p={p} match={match} onOpen={onOpen} />
                ))}
              </tbody>
            </table>
          </div>
          {rows.length > visible.length ? (
            <div className="list-more">
              <button type="button" className="btn btn--ghost" onClick={() => setShowAll(true)} data-testid="possessions-more">
                Show all {rows.length} possessions
              </button>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function PossessionRow({ p, match, onOpen }: { p: PossessionSummary; match: MatchSummary; onOpen: (pid: string) => void }) {
  const clock = fmtClock(matchClockSeconds(p.start_time_s, p.period, match.periods, match.fps));
  const first = titleCase(p.first_type);
  const last = [titleCase(p.last_type), p.last_subtype ? `(${titleCase(p.last_subtype)})` : ""].filter(Boolean).join(" ");
  const teams = matchTeams(match);
  const label = `${teamName(p.team, teams)} possession, period ${p.period}, ${clock}, ${p.n_passes} ${p.n_passes === 1 ? "pass" : "passes"}${p.has_goal ? ", goal" : p.has_shot ? ", shot" : ""}`;
  return (
    <tr className="row-link" onClick={() => onOpen(p.possession_id)}>
      <td>
        <a
          className="team-chip-link"
          href={routeHash({ name: "lab", possessionId: p.possession_id })}
          aria-label={`Open ${label}`}
          onClick={(e) => e.stopPropagation()}
        >
          <span className={`team-chip team-chip--${p.team}`}>{teamName(p.team, teams)}</span>
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
