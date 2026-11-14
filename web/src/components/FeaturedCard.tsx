import type { DefensiveShadowOut, MatchSummary, PlayerState, ScenarioResponse } from "../api/types";
import { featuredFacts, featuredHref, type FeaturedScenario } from "../lib/featured";
import { fmtMetres, fmtProb, fmtSigned } from "../lib/format";
import { useAsync } from "../lib/hooks";
import { featuredScenario, featuredShadow } from "../lib/featuredLive";
import { hasTeamNames, matchTeams, matchTitle, teamName, type TeamNames } from "../lib/teams";
import { ProvBadge } from "./ProvBadge";
import { StatusBlock } from "./StatusBlock";

function label(id: string, players: PlayerState[], t: TeamNames): string {
  const p = players.find((q) => q.player_id === id);
  return p ? `#${p.jersey} ${teamName(p.team, t)}` : id;
}

/**
 * One featured situation. The story is curated text; every number is read from the API
 * (the same scenario request the lab makes, and the Defensive Shadow at that frame).
 */
export function FeaturedCard({
  f,
  match,
  testId,
}: {
  f: FeaturedScenario;
  match: MatchSummary | null;
  testId: string;
}) {
  const teams = matchTeams(match);
  const live = useAsync<ScenarioResponse>(`featured:${f.id}`, () => featuredScenario(f));
  // the shadow is an optional extra line, fetched after the scenarios; the card stands without it
  const sh = useAsync<DefensiveShadowOut | null>(`featured-shadow:${f.id}`, () => featuredShadow(f));
  const facts = live.data ? featuredFacts(f, live.data, sh.data ?? null, match) : null;
  const shadowPending = sh.status === "loading" || sh.status === "idle";
  const players = live.data?.observed.players ?? [];
  const who = (id: string) => label(id, players, teams);
  const titleId = `featured-${f.id}-title`;
  const named = match && hasTeamNames(teams);

  return (
    <article className="feature-card" aria-labelledby={titleId} data-testid={`featured-card-${f.id}`}>
      <p className="eyebrow feature-kicker">
        {f.source}
        {facts?.clock ? <span className="tabular"> · {facts.clock}</span> : null}
      </p>
      <h3 id={titleId} className="feature-title">
        {f.title}
      </h3>
      {named ? <p className="small muted feature-match">{matchTitle(match)}</p> : null}
      <p className="feature-story">{f.story(teams)}</p>

      <div className="feature-live" aria-live="polite" aria-busy={live.status === "loading"}>
        {live.status === "error" ? (
          <StatusBlock
            kind="error"
            title="Live numbers unavailable"
            error={live.error}
            subject="this featured situation"
            onRetry={live.reload}
            compact
          />
        ) : !facts ? (
          <div className="feature-facts feature-facts--loading" data-testid="featured-loading">
            <span className="skeleton" />
            <span className="skeleton" />
            <span className="sr-only">Computing the live numbers</span>
          </div>
        ) : (
          <dl className="feature-facts" data-testid="featured-facts">
            {shadowPending ? (
              <div className="fact" data-testid="featured-shadow-loading">
                <dt>Defensive Shadow</dt>
                <dd>
                  <span className="skeleton skeleton--inline" />
                </dd>
              </div>
            ) : facts.shadow !== null ? (
              <div className="fact">
                <dt>
                  Defensive Shadow of {who(f.defender)} <span className="badge badge--experimental">experimental</span>
                </dt>
                <dd>
                  <span className="fact-value tabular">{fmtProb(facts.shadow)}</span>{" "}
                  {facts.nextShadow !== null ? (
                    <span className="small muted">next highest defender {fmtProb(facts.nextShadow)}</span>
                  ) : null}{" "}
                  <ProvBadge prov="estimated" />
                </dd>
              </div>
            ) : null}
            {facts.pass ? (
              <div className="fact">
                <dt>
                  Pass played, {who(f.passer)} → {who(f.receiver)}: estimated completion
                </dt>
                <dd>
                  <span className="fact-value tabular">
                    {fmtProb(facts.pass.observed_p)} <span aria-hidden="true">→</span>
                    <span className="sr-only">to</span> {fmtProb(facts.pass.hypothetical_p)}
                  </span>{" "}
                  <span className={`tabular ${(facts.pass.delta ?? 0) > 0 ? "pos" : "neg"}`}>{fmtSigned(facts.pass.delta)}</span>{" "}
                  <ProvBadge prov="estimated" />
                  <span className="fact-sub small muted">
                    observed, then with {who(f.defender)} moved {fmtMetres(facts.displacement_m)}{" "}
                    <ProvBadge prov="hypothetical" />
                    {facts.agreeing !== null ? ` · ${facts.agreeing} of ${facts.ensembleSize} model settings agree on the direction` : ""}
                    {facts.supported ? " · physically plausible edit" : " · the edit fails a feasibility check"}
                  </span>
                </dd>
              </div>
            ) : null}
            {facts.estimated > 0 ? (
              <div className="fact">
                <dt>Player positions at this frame</dt>
                <dd>
                  <span className="fact-value tabular">{facts.shown - facts.estimated}</span>{" "}
                  <ProvBadge prov="observed" /> <span className="fact-value tabular">{facts.estimated}</span>{" "}
                  <ProvBadge prov="estimated">estimated (off camera)</ProvBadge>
                </dd>
              </div>
            ) : null}
          </dl>
        )}
      </div>

      <div className="feature-actions">
        <a className="btn btn--primary" href={featuredHref(f, "tour")} data-testid={`tour-link-${f.id}`}>
          Take the guided tour <span aria-hidden="true">→</span>
        </a>
        <a className="btn" href={featuredHref(f, "scenario")} data-testid={testId}>
          Open the what-if directly
        </a>
      </div>
    </article>
  );
}
