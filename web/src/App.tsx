import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api/client";
import type { MatchSummary, PossessionSummary } from "./api/types";
import { Landing } from "./components/Landing";
import { Lab } from "./components/Lab";
import { SourceAttribution } from "./components/SourceAttribution";
import { StatusBlock } from "./components/StatusBlock";
import { providerInfo } from "./lib/providers";
import { matchOfPossession, parseRoute, routeHash, useAsync, useHash } from "./lib/hooks";
import type { RouteMove } from "./lib/hooks";

export function App() {
  const [hash, go] = useHash();
  const route = parseRoute(hash);
  const matches = useAsync<MatchSummary[]>("matches", (signal) => api.matches(signal));

  const matchId =
    route.name === "lab"
      ? matchOfPossession(route.possessionId)
      : route.name === "at"
        ? route.matchId
        : route.matchId ?? matches.data?.[0]?.match_id ?? null;
  // On lab routes, fetch the one match summary (milliseconds) instead of waiting for the full list,
  // so the clock, top bar and attribution are right from the first paint.
  const single = useAsync<MatchSummary>(
    route.name === "lab" && !matches.data && matchId ? `match:${matchId}` : null,
    (signal) => api.match(matchId ?? "", signal),
  );
  const match = useMemo(
    () => matches.data?.find((m) => m.match_id === matchId) ?? (single.data?.match_id === matchId ? single.data : null),
    [matches.data, matchId, single.data],
  );

  // Coming back from the lab (its "Possessions" button or the browser's Back), or opening a
  // #/match/<id> link directly, the landing brings the possession list (or the "match not found"
  // notice) into view instead of opening at the top of the page.
  const [revealList, setRevealList] = useState(() => route.name === "home" && route.matchId !== null);
  const prevRoute = useRef(route.name);
  useEffect(() => {
    if (route.name === "home" && prevRoute.current !== "home") setRevealList(true);
    prevRoute.current = route.name;
  }, [route.name]);
  const listRevealed = useCallback(() => setRevealList(false), []);

  const openPossession = useCallback((pid: string) => go(routeHash({ name: "lab", possessionId: pid })), [go]);
  const selectMatch = useCallback(
    (mid: string) => {
      setRevealList(false);
      go(routeHash({ name: "home", matchId: mid }));
    },
    [go],
  );
  const goHome = useCallback(() => go(routeHash({ name: "home", matchId })), [go, matchId]);

  useEffect(() => {
    // both lab routes (#/lab/… and the #/at/… links of the featured situations and the tour)
    document.title = route.name === "home" ? "FootyMind · The Invisible Game" : "Lab · FootyMind";
  }, [route.name]);

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <a className="brand" href={routeHash({ name: "home", matchId })} aria-label="FootyMind home">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">FootyMind</span>
          <span className="brand-sub">The Invisible Game</span>
        </a>
        <nav className="topbar-meta" aria-label="Data source">
          {match ? (
            <span className="topbar-source">
              Data: <a href={match.source.url} target="_blank" rel="noreferrer">{match.source.name}</a>
              {providerInfo(match.source).anonymised ? " (anonymised)" : ""}
            </span>
          ) : null}
        </nav>
      </header>

      <main id="main" className="main">
        {route.name === "at" ? (
          <AtFrame
            key={`${route.matchId}@${route.frame}`}
            matchId={route.matchId}
            frame={route.frame}
            moves={route.moves}
            response={route.response}
            tour={route.tour}
            match={match}
            matchMissing={!!matches.data && !match}
            onBack={goHome}
          />
        ) : route.name === "lab" ? (
          // The lab needs only the possession replay; it does not wait for the match list
          // (the match clock falls back to provider time until the list arrives).
          <Lab
            key={route.possessionId}
            possessionId={route.possessionId}
            match={match}
            onBack={goHome}
            initialFrame={route.frame}
            initialMoves={route.moves}
            initialResponse={route.response}
            tour={route.tour}
          />
        ) : matches.status === "error" && !matches.data ? (
          <StatusBlock
            kind="error"
            title="Could not load matches"
            error={matches.error}
            onRetry={matches.reload}
            pageHeading
          />
        ) : !matches.data ? (
          <StatusBlock kind="loading" title="Loading matches" pageHeading />
        ) : matches.data.length === 0 ? (
          <StatusBlock
            kind="empty"
            pageHeading
            title="No processed matches"
            detail="Run the ingestion pipeline (footymind ingest) and restart the API."
          />
        ) : (
          <Landing
            matches={matches.data}
            match={match}
            onSelectMatch={selectMatch}
            onOpenPossession={openPossession}
            requestedMatchId={route.name === "home" ? route.matchId : null}
            revealList={revealList}
            onRevealed={listRevealed}
          />
        )}
      </main>

      <footer className="footer">{match ? <SourceAttribution source={match.source} /> : null}</footer>
    </div>
  );
}

/** Resolves a stable (match, frame) deep link to its possession, then opens the lab there. */
function AtFrame({
  matchId,
  frame,
  moves,
  response,
  tour,
  match,
  matchMissing,
  onBack,
}: {
  matchId: string;
  frame: number;
  moves?: RouteMove[];
  response?: string;
  tour?: string;
  match: MatchSummary | null;
  /** the match list has loaded and does not contain this match */
  matchMissing: boolean;
  onBack: () => void;
}) {
  const pos = useAsync<PossessionSummary>(`at:${matchId}@${frame}`, (signal) =>
    api.framePossession(matchId, frame, signal),
  );
  if (pos.status === "error") {
    return (
      <div className="lab lab--status">
        <StatusBlock
          kind="error"
          title="Could not open this moment"
          error={pos.error}
          subject={matchMissing ? `the match "${matchId}" on this server` : `a possession at frame ${frame} of this match`}
          onRetry={pos.reload}
          pageHeading
          action={
            <button type="button" className="btn btn--sm" onClick={onBack}>
              Browse possessions
            </button>
          }
        />
      </div>
    );
  }
  if (!pos.data) return <StatusBlock kind="loading" title="Finding the possession" pageHeading />;
  return (
    <Lab
      possessionId={pos.data.possession_id}
      match={match}
      onBack={onBack}
      initialFrame={frame}
      initialMoves={moves}
      initialResponse={response}
      tour={tour}
    />
  );
}
