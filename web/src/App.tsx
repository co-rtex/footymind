import { useCallback, useEffect, useMemo } from "react";
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

  const openPossession = useCallback((pid: string) => go(routeHash({ name: "lab", possessionId: pid })), [go]);
  const selectMatch = useCallback((mid: string) => go(routeHash({ name: "home", matchId: mid })), [go]);
  const goHome = useCallback(() => go(routeHash({ name: "home", matchId })), [go, matchId]);

  useEffect(() => {
    document.title = route.name === "lab" ? "Lab · FootyMind" : "FootyMind · The Invisible Game";
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
            match={match}
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
          />
        ) : matches.status === "error" && !matches.data ? (
          <StatusBlock kind="error" title="Could not load matches" error={matches.error} onRetry={matches.reload} />
        ) : !matches.data ? (
          <StatusBlock kind="loading" title="Loading matches" />
        ) : matches.data.length === 0 ? (
          <StatusBlock
            kind="empty"
            title="No processed matches"
            detail="Run the ingestion pipeline (footymind ingest) and restart the API."
          />
        ) : (
          <Landing
            matches={matches.data}
            match={match}
            onSelectMatch={selectMatch}
            onOpenPossession={openPossession}
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
  match,
  onBack,
}: {
  matchId: string;
  frame: number;
  moves?: RouteMove[];
  match: MatchSummary | null;
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
          subject={`a possession at frame ${frame} of this match`}
          onRetry={pos.reload}
          action={
            <button type="button" className="btn btn--sm" onClick={onBack}>
              Browse possessions
            </button>
          }
        />
      </div>
    );
  }
  if (!pos.data) return <StatusBlock kind="loading" title="Finding the possession" />;
  return (
    <Lab
      possessionId={pos.data.possession_id}
      match={match}
      onBack={onBack}
      initialFrame={frame}
      initialMoves={moves}
    />
  );
}
