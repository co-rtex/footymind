import { useCallback, useEffect, useMemo } from "react";
import { api } from "./api/client";
import type { MatchSummary } from "./api/types";
import { Landing } from "./components/Landing";
import { Lab } from "./components/Lab";
import { SourceAttribution } from "./components/SourceAttribution";
import { StatusBlock } from "./components/StatusBlock";
import { matchOfPossession, parseRoute, routeHash, useAsync, useHash } from "./lib/hooks";

export function App() {
  const [hash, go] = useHash();
  const route = parseRoute(hash);
  const matches = useAsync<MatchSummary[]>("matches", (signal) => api.matches(signal));

  const matchId =
    route.name === "lab" ? matchOfPossession(route.possessionId) : route.matchId ?? matches.data?.[0]?.match_id ?? null;
  const match = useMemo(() => matches.data?.find((m) => m.match_id === matchId) ?? null, [matches.data, matchId]);

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
              Data: <a href={match.source.url} target="_blank" rel="noreferrer">{match.source.name}</a> (anonymised)
            </span>
          ) : null}
        </nav>
      </header>

      <main id="main" className="main">
        {matches.status === "error" && !matches.data ? (
          <StatusBlock kind="error" title="Could not load matches" detail={matches.error.message} onRetry={matches.reload} />
        ) : !matches.data ? (
          <StatusBlock kind="loading" title="Loading matches" />
        ) : matches.data.length === 0 ? (
          <StatusBlock
            kind="empty"
            title="No processed matches"
            detail="Run the ingestion pipeline (footymind ingest) and restart the API."
          />
        ) : route.name === "lab" ? (
          <Lab key={route.possessionId} possessionId={route.possessionId} match={match} onBack={goHome} />
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
