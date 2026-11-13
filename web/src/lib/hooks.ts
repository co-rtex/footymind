import { useCallback, useEffect, useRef, useState } from "react";

export type AsyncState<T> =
  | { status: "idle"; data: null; error: null }
  | { status: "loading"; data: T | null; error: null }
  | { status: "success"; data: T; error: null }
  | { status: "error"; data: T | null; error: Error };

const IDLE = { status: "idle", data: null, error: null } as const;

/**
 * Run an abortable async loader whenever `key` changes. A null key means
 * "nothing to load". Stale responses are discarded via AbortController.
 */
export function useAsync<T>(
  key: string | null,
  loader: (signal: AbortSignal) => Promise<T>,
): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>(IDLE);
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    if (key === null) {
      setState(IDLE);
      return;
    }
    const ac = new AbortController();
    setState((s) => ({ status: "loading", data: s.data, error: null }));
    loaderRef.current(ac.signal).then(
      (data) => {
        if (!ac.signal.aborted) setState({ status: "success", data, error: null });
      },
      (error: unknown) => {
        if (ac.signal.aborted) return;
        setState((s) => ({ status: "error", data: s.data, error: error instanceof Error ? error : new Error(String(error)) }));
      },
    );
    return () => ac.abort();
  }, [key, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}

/** Current location hash, kept in React state. */
export function useHash(): [string, (h: string) => void] {
  const [hash, setHash] = useState(() => (typeof window === "undefined" ? "" : window.location.hash));
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((h: string) => {
    if (window.location.hash !== h) window.location.hash = h;
  }, []);
  return [hash, go];
}

/** A deep-linkable scenario move: player relocated to canonical (x, y) metres. */
export interface RouteMove {
  player_id: string;
  x: number;
  y: number;
}

export type Route =
  | { name: "home"; matchId: string | null }
  | { name: "lab"; possessionId: string; frame?: number; moves?: RouteMove[] };

/**
 * Hash routes:
 *   #/match/<matchId>
 *   #/lab/<possessionId>[?f=<frame>][&move=<player_id>,<x>,<y>]...
 * Invalid optional parameters are ignored, never guessed.
 */
export function parseRoute(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [path = "", query = ""] = h.split("?", 2);
  const [head, ...rest] = path.split("/");
  const arg = rest.join("/");
  if (head === "lab" && arg) {
    const route: Route = { name: "lab", possessionId: decodeURIComponent(arg) };
    const params = new URLSearchParams(query);
    const f = Number(params.get("f"));
    if (params.has("f") && Number.isInteger(f) && f > 0) route.frame = f;
    const moves: RouteMove[] = [];
    for (const m of params.getAll("move")) {
      const [pid, xs, ys] = m.split(",");
      const x = Number(xs);
      const y = Number(ys);
      if (pid && Number.isFinite(x) && Number.isFinite(y) && Math.abs(x) <= 60 && Math.abs(y) <= 40) {
        moves.push({ player_id: pid, x, y });
      }
    }
    if (moves.length) route.moves = moves.slice(0, 4);
    return route;
  }
  if (head === "match" && arg) return { name: "home", matchId: decodeURIComponent(arg) };
  return { name: "home", matchId: null };
}

export function routeHash(r: Route): string {
  if (r.name === "lab") {
    const params = new URLSearchParams();
    if (r.frame !== undefined) params.set("f", String(r.frame));
    for (const m of r.moves ?? []) params.append("move", `${m.player_id},${m.x},${m.y}`);
    const q = params.toString();
    return `#/lab/${encodeURIComponent(r.possessionId)}${q ? `?${q.replace(/%2C/g, ",")}` : ""}`;
  }
  return r.matchId ? `#/match/${encodeURIComponent(r.matchId)}` : "#/";
}

/** Match id embedded in a possession id ("<match>:pNNNN"); split on the first colon like the backend. */
export function matchOfPossession(pid: string): string {
  const i = pid.indexOf(":");
  return i > 0 ? pid.slice(0, i) : pid;
}
