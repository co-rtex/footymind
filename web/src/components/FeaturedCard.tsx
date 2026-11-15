import type { DefensiveShadowOut, MatchSummary, OptionDeltaOut, PlayerState } from "../api/types";
import { featuredFacts, featuredHref, type FeaturedFacts, type FeaturedScenario, type ResponseFact } from "../lib/featured";
import { fmtMetres, fmtProb, fmtSigned } from "../lib/format";
import { useAsync } from "../lib/hooks";
import { featuredScenarios, featuredShadow } from "../lib/featuredLive";
import { policyLabel } from "../lib/policies";
import { PolicyBadge } from "./PolicyBadge";
import { hasTeamNames, matchTeams, matchTitle, teamName, type TeamNames } from "../lib/teams";
import { ProvBadge } from "./ProvBadge";
import { StatusBlock } from "./StatusBlock";

function label(id: string, players: PlayerState[], t: TeamNames): string {
  const p = players.find((q) => q.player_id === id);
  return p ? `#${p.jersey} ${teamName(p.team, t)}` : id;
}

function signCls(d: number | null): string {
  return (d ?? 0) > 0 ? "pos" : "neg";
}

/** "0.45 → 0.76", with "to" for screen readers. */
function Transition({ pass }: { pass: OptionDeltaOut }) {
  return (
    <>
      {fmtProb(pass.observed_p)} <span aria-hidden="true">→</span>
      <span className="sr-only">to</span> {fmtProb(pass.hypothetical_p)}
    </>
  );
}

/** "24 of 24 model settings agree on the direction under each assumption", from both results. */
function agreementText(facts: FeaturedFacts, r: ResponseFact, name: string): string | null {
  if (facts.agreeing === null || r.agreeing === null) return null;
  if (facts.agreeing === r.agreeing && facts.ensembleSize === r.ensembleSize) {
    return `${facts.agreeing} of ${facts.ensembleSize} model settings agree on the direction under each assumption`;
  }
  return `model settings agreeing on the direction: ${facts.agreeing} of ${facts.ensembleSize} static, ${r.agreeing} of ${r.ensembleSize} ${name}`;
}

/** Static defenders vs the situation's defender response, for the pass that was played. */
function ResponseCompare({
  f,
  facts,
  r,
  who,
}: {
  f: FeaturedScenario;
  facts: FeaturedFacts;
  r: ResponseFact;
  who: (id: string) => string;
}) {
  const name = policyLabel(r.policy);
  const agree = agreementText(facts, r, name);
  const rows = [
    { key: "static", label: "Defenders stand still", pass: facts.pass, info: null },
    { key: "response", label: name, pass: r.pass, info: r.info ?? { name: r.policy, validated: false, status: null } },
  ];
  return (
    <div className="fact" data-testid="featured-response">
      <dt>
        Pass played, {who(f.passer)} → {who(f.receiver)}: estimated completion, with {who(f.moved)} moved{" "}
        {fmtMetres(facts.displacement_m)} <ProvBadge prov="hypothetical" />
      </dt>
      <dd>
        <table className="fact-compare">
          <caption className="sr-only">
            Estimated completion of the pass that was played, observed and with the move, under two assumptions about
            the defence
          </caption>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} data-testid={`featured-response-${row.key}`}>
                <th scope="row">
                  {row.label}
                  {row.info ? (
                    <>
                      {" "}
                      <PolicyBadge info={row.info} testId="featured-policy-badge" />
                    </>
                  ) : null}
                </th>
                <td className="tabular fact-compare-value">{row.pass ? <Transition pass={row.pass} /> : "n/a"}</td>
                <td className={`tabular num ${signCls(row.pass?.delta ?? null)}`}>{fmtSigned(row.pass?.delta)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fact-verdict" data-testid="featured-response-verdict">
          {r.flips === true
            ? "The direction of the change depends on the assumption about the defence. "
            : r.flips === false
              ? "Both assumptions point the same way. "
              : ""}
          <ProvBadge prov="estimated">model estimates</ProvBadge>
        </p>
        <span className="fact-sub small muted">
          {r.simulated.length > 0
            ? r.simulated.map((m, i) => (
                <span key={m.player_id} data-testid="featured-simulated-move">
                  {i > 0 ? ", " : ""}
                  {who(m.player_id)} moved {fmtMetres(m.distance_m)} by the {name} response{" "}
                  <ProvBadge prov="simulated" />
                </span>
              ))
            : `The ${name} response moved no defender.`}
          {agree ? ` · ${agree}` : ""}
          {facts.supported && r.supported ? " · physically plausible edits" : " · an edit fails a feasibility check"}
        </span>
      </dd>
    </div>
  );
}

/**
 * One featured situation. The story is curated text; every number is read from the API
 * (the same scenario requests the lab makes and, where the situation shows it, the Defensive
 * Shadow at that frame).
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
  const live = useAsync(`featured:${f.id}`, () => featuredScenarios(f));
  // the shadow is an optional extra line, fetched after the scenarios; the card stands without it
  const sh = useAsync<DefensiveShadowOut | null>(f.showShadow ? `featured-shadow:${f.id}` : null, () => featuredShadow(f));
  const facts = live.data
    ? featuredFacts(f, { scenario: live.data.scenario, response: live.data.response, shadow: sh.data, match })
    : null;
  const shadowPending = f.showShadow && (sh.status === "loading" || sh.status === "idle");
  const players = live.data?.scenario.observed.players ?? [];
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
            ) : facts.shadow ? (
              <div className="fact">
                <dt>
                  Defensive Shadow of {who(f.moved)} <span className="badge badge--experimental">experimental</span>
                </dt>
                <dd>
                  <span className="fact-value tabular">{fmtProb(facts.shadow.value)}</span>{" "}
                  {facts.shadow.next !== null ? (
                    <span className="small muted">next highest defender {fmtProb(facts.shadow.next)}</span>
                  ) : null}{" "}
                  <ProvBadge prov="estimated" />
                </dd>
              </div>
            ) : null}
            {facts.response ? (
              <ResponseCompare f={f} facts={facts} r={facts.response} who={who} />
            ) : facts.pass ? (
              <div className="fact">
                <dt>
                  Pass played, {who(f.passer)} → {who(f.receiver)}: estimated completion
                </dt>
                <dd>
                  <span className="fact-value tabular">
                    <Transition pass={facts.pass} />
                  </span>{" "}
                  <span className={`tabular ${signCls(facts.pass.delta)}`}>{fmtSigned(facts.pass.delta)}</span>{" "}
                  <ProvBadge prov="estimated" />
                  <span className="fact-sub small muted">
                    observed, then with {who(f.moved)} moved {fmtMetres(facts.displacement_m)}{" "}
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
