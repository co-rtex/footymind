import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ReplayData } from "../api/parse";
import type {
  DefensiveShadowOut,
  FrameAnalysisOut,
  InterventionIn,
  MatchSummary,
  PlayerOut,
  PolicyInfo,
  ResponsePolicy,
  ScenarioRequest,
  ScenarioResponse,
  ValueModel,
} from "../api/types";
import { MAX_INTERVENTIONS } from "../api/types";
import { fmtSigned, playerLabel, speedOf, titleCase } from "../lib/format";
import { matchTeams, matchTitle, labelledPlayers, teamName } from "../lib/teams";
import { matchOfPossession, useAsync } from "../lib/hooks";
import { finiteDiffSpeed, valueAt } from "../lib/interp";
import { isNegligible } from "../lib/agreement";
import { FALLBACK_POLICIES, orderPolicies, STATIC_POLICY } from "../lib/policies";
import { providerInfo } from "../lib/providers";
import { Playback } from "../lib/playback";
import { clockAt, eventLabel, indexOfFrame, periodAt } from "../lib/replayIndex";
import type { DrawnPlayer, Layers, PitchRenderer, ScenarioView } from "../render/renderer";
import { AnalysisPanel, PASS_TABLE_ID } from "./AnalysisPanel";
import { Legend } from "./Legend";
import { PitchCanvas, type DragInit } from "./PitchCanvas";
import { PlayerInspector, type InspectedPlayer } from "./PlayerInspector";
import { ScenarioPanel } from "./ScenarioPanel";
import { ShadowPanel } from "./ShadowPanel";
import { SimilarPanel } from "./SimilarPanel";
import { WhyPanel } from "./WhyPanel";
import { Segmented } from "./Segmented";
import { StatusBlock } from "./StatusBlock";
import { Transport } from "./Transport";

interface Props {
  possessionId: string;
  match: MatchSummary | null;
  onBack: () => void;
  /** Deep link: provider frame to open paused at (must lie inside the replay window). */
  initialFrame?: number;
  /** Deep link: scenario moves to apply on open (opens in edit mode, difference view). */
  initialMoves?: InterventionIn[];
}

export function Lab({ possessionId, match, onBack, initialFrame, initialMoves }: Props) {
  const replay = useAsync<ReplayData>(`replay:${possessionId}`, (signal) => api.possessionReplay(possessionId, signal));
  // The match clock needs period start frames. Fetch this match's summary directly (milliseconds)
  // rather than waiting for the full match list, and never show a provider-time clock in its place.
  const matchId = matchOfPossession(possessionId);
  const own = useAsync<MatchSummary>(match ? null : `match:${matchId}`, (signal) => api.match(matchId, signal));
  const summary = match ?? own.data;
  if (replay.status === "error") {
    return (
      <div className="lab lab--status">
        <BackButton onBack={onBack} />
        <StatusBlock
          kind="error"
          title="Could not load this possession"
          error={replay.error}
          subject="this possession"
          onRetry={replay.reload}
          action={
            <button type="button" className="btn btn--sm" onClick={onBack}>
              Browse possessions
            </button>
          }
        />
      </div>
    );
  }
  if (!replay.data || (!summary && own.status !== "error")) {
    return (
      <div className="lab lab--status">
        <BackButton onBack={onBack} />
        <StatusBlock kind="loading" title="Loading tracking replay" />
      </div>
    );
  }
  return (
    <LabView
      replay={replay.data}
      match={summary}
      onBack={onBack}
      initialFrame={initialFrame}
      initialMoves={initialMoves}
    />
  );
}

function BackButton({ onBack }: { onBack: () => void }) {
  return (
    <button type="button" className="btn btn--ghost btn--sm back" onClick={onBack}>
      <span aria-hidden="true">←</span> Possessions
    </button>
  );
}

type Mode = "analyse" | "edit";
type Loadable<T> = { status: "idle" | "loading" | "success" | "error"; data: T | null; error: Error | null };
const IDLE: Loadable<never> = { status: "idle", data: null, error: null };
/** A loadable that also records which request (serialised) its data or error belongs to. */
type KeyedLoadable<T> = Loadable<T> & { key: string | null };
const KEYED_IDLE: KeyedLoadable<never> = { ...IDLE, key: null };

/**
 * Run (or read from cache) the scenario request serialised in `key`. `keepStale` keeps the
 * previous result visible (marked loading) while a new one is computed.
 */
function useScenarioRequest(
  key: string | null,
  cache: Map<string, ScenarioResponse>,
  set: (f: (s: KeyedLoadable<ScenarioResponse>) => KeyedLoadable<ScenarioResponse>) => void,
  nonce: number,
  keepStale: boolean,
): void {
  useEffect(() => {
    if (key === null) {
      set(() => KEYED_IDLE);
      return;
    }
    const hit = cache.get(key);
    if (hit) {
      set(() => ({ status: "success", data: hit, error: null, key }));
      return;
    }
    const ac = new AbortController();
    set((s) => ({ status: "loading", data: keepStale ? s.data : null, error: null, key: keepStale ? s.key : null }));
    api.scenario(JSON.parse(key) as ScenarioRequest, ac.signal).then(
      (d) => {
        cachePut(cache, key, d);
        if (!ac.signal.aborted) set(() => ({ status: "success", data: d, error: null, key }));
      },
      (e: unknown) => {
        if (ac.signal.aborted) return;
        const error = e instanceof Error ? e : new Error(String(e));
        set((s) => ({ status: "error", data: keepStale ? s.data : null, error, key }));
      },
    );
    return () => ac.abort();
  }, [key, cache, set, nonce, keepStale]);
}
const ANALYSIS_DEBOUNCE_MS = 160;
const SHADOW_DEBOUNCE_MS = 250;
const HORIZON_S = 1.5;
const ENSEMBLE = 24;
const NO_PENDING: ReadonlyMap<string, { x: number; y: number }> = new Map();
const NO_TRACE: readonly string[] = [];
const SCENARIO_CACHE_MAX = 24;

/**
 * Response policies per match, fetched once per session. A backend without the endpoint (404)
 * is remembered, so older backends cost one failed request, not one per edit-mode entry.
 */
const policyRequests = new Map<string, Promise<PolicyInfo[]>>();
function loadPolicies(matchId: string): Promise<PolicyInfo[]> {
  let p = policyRequests.get(matchId);
  if (!p) {
    p = api.responsePolicies(matchId).catch((e: unknown) => {
      if (!(e instanceof ApiError && e.status === 404)) policyRequests.delete(matchId);
      return FALLBACK_POLICIES;
    });
    policyRequests.set(matchId, p);
  }
  return p;
}

function scenarioRequest(
  matchId: string,
  frame: number,
  interventions: InterventionIn[],
  policy: ResponsePolicy,
  valueModel: ValueModel,
): ScenarioRequest {
  return {
    match_id: matchId,
    frame,
    interventions,
    response_policy: policy,
    horizon_s: HORIZON_S,
    ensemble: ENSEMBLE,
    value_model: valueModel,
  };
}

/** Scenario results keyed by request, so switching policy or view never recomputes a known edit. */
function cachePut(cache: Map<string, ScenarioResponse>, key: string, v: ScenarioResponse): void {
  cache.set(key, v);
  while (cache.size > SCENARIO_CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

function LabView({
  replay,
  match,
  onBack,
  initialFrame,
  initialMoves,
}: {
  replay: ReplayData;
  match: MatchSummary | null;
  onBack: () => void;
  initialFrame?: number;
  initialMoves?: InterventionIn[];
}) {
  // Team display names (real names for SkillCorner, Home/Away for anonymised Metrica).
  const teams = useMemo(() => matchTeams(match), [match]);
  const players: Map<string, PlayerOut> = useMemo(() => labelledPlayers(replay.players, teams), [replay, teams]);
  const startIdx = useMemo(() => {
    const first = replay.frame[0];
    const last = replay.frame[replay.frame.length - 1];
    if (initialFrame !== undefined && first !== undefined && last !== undefined && initialFrame >= first && initialFrame <= last) {
      return indexOfFrame(replay.frame, initialFrame);
    }
    return replay.possession ? indexOfFrame(replay.frame, replay.possession.start_frame) : 0;
  }, [replay, initialFrame]);
  // Deep-linked moves only apply to players present in this replay.
  const linkedMoves = useMemo(
    () => (initialMoves ?? []).filter((m) => replay.players.some((p) => p.player_id === m.player_id)),
    [initialMoves, replay],
  );
  const playback = useMemo(() => {
    const pb = new Playback(replay.nFrames, replay.fps, replay.step);
    pb.t = startIdx;
    return pb;
  }, [replay, startIdx]);
  useEffect(() => () => playback.destroy(), [playback]);

  const rendererRef = useRef<PitchRenderer | null>(null);
  const [rendererReady, setRendererReady] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [index, setIndex] = useState(startIdx);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState<Mode>(linkedMoves.length ? "edit" : "analyse");
  const [layers, setLayers] = useState<Layers>({ control: true, passes: true, pressure: true, trails: true, shadow: false });
  const [view, setView] = useState<ScenarioView>(linkedMoves.length ? "difference" : "observed");
  const [selected, setSelected] = useState<string | null>(null);
  /** bumped when a player is chosen from the Player menu in edit mode, to focus the move form */
  const [focusMove, setFocusMove] = useState(0);
  const [highlight, setHighlight] = useState<string | null>(null);
  /** players named by the focused "why this moment" statement */
  const [trace, setTrace] = useState<readonly string[]>(NO_TRACE);
  const [interventions, setInterventions] = useState<InterventionIn[]>(linkedMoves);
  const [policy, setPolicy] = useState<ResponsePolicy>("none");
  const [valueModel, setValueModel] = useState<ValueModel>("option_model");
  const [dragWarn, setDragWarn] = useState<{ outside: boolean | null; id: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // ---- playback -> renderer (imperative, per animation frame) ----
  useEffect(() => {
    const offTick = playback.onTick((t) => {
      const r = rendererRef.current;
      if (r) {
        r.setScene({ t }, false);
        r.drawNow();
      }
      if (!playback.playing) setIndex(Math.round(t));
    });
    const offState = playback.onState((p) => {
      setPlaying(p);
      rendererRef.current?.setScene({ playing: p });
      if (!p) setIndex(Math.round(playback.t));
    });
    return () => {
      offTick();
      offState();
    };
  }, [playback]);

  useEffect(() => playback.setSpeed(speed), [playback, speed]);

  // ---- frame analysis: fetched when paused, debounced while scrubbing, cached by frame ----
  const cache = useRef(new Map<number, FrameAnalysisOut>());
  const [an, setAn] = useState<Loadable<FrameAnalysisOut>>(IDLE);
  const [anNonce, setAnNonce] = useState(0);
  const frame = replay.frame[index] as number;
  useEffect(() => {
    if (playing) return;
    const hit = cache.current.get(frame);
    if (hit) {
      setAn({ status: "success", data: hit, error: null });
      return;
    }
    setAn((s) => ({ status: "loading", data: s.data, error: null }));
    const ac = new AbortController();
    const timer = setTimeout(() => {
      api.frameAnalysis(replay.matchId, frame, ac.signal).then(
        (d) => {
          cache.current.set(frame, d);
          if (!ac.signal.aborted) setAn({ status: "success", data: d, error: null });
        },
        (e: unknown) => {
          if (!ac.signal.aborted) setAn((s) => ({ status: "error", data: s.data, error: e instanceof Error ? e : new Error(String(e)) }));
        },
      );
    }, ANALYSIS_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [playing, frame, replay.matchId, anNonce]);
  const analysis = an.data && an.data.frame === frame ? an.data : null;

  // ---- defensive shadow (experimental): lazy, paused + analyse mode only, debounced, cached ----
  // Each frame costs about 0.4 s on the backend, so it is never fetched during playback.
  const shadowAvailable = mode === "analyse" && !playing;
  const shadowWanted = layers.shadow && shadowAvailable;
  const shadowCache = useRef(new Map<number, { data: DefensiveShadowOut } | { unavailable: string }>());
  const [sh, setSh] = useState<Loadable<DefensiveShadowOut> & { unavailable: string | null }>({ ...IDLE, unavailable: null });
  const [shNonce, setShNonce] = useState(0);
  useEffect(() => {
    if (!shadowWanted) return;
    const hit = shadowCache.current.get(frame);
    if (hit) {
      setSh(
        "data" in hit
          ? { status: "success", data: hit.data, error: null, unavailable: null }
          : { status: "success", data: null, error: null, unavailable: hit.unavailable },
      );
      return;
    }
    setSh((s) => ({ status: "loading", data: s.data, error: null, unavailable: null }));
    const ac = new AbortController();
    const timer = setTimeout(() => {
      api.defensiveShadow(replay.matchId, frame, { horizonS: 1.0, valueModel: "option_model" }, ac.signal).then(
        (d) => {
          shadowCache.current.set(frame, { data: d });
          if (!ac.signal.aborted) setSh({ status: "success", data: d, error: null, unavailable: null });
        },
        (e: unknown) => {
          if (ac.signal.aborted) return;
          // 422 is deterministic (no ball / no possession): cache it as "unavailable", not as an error.
          if (e instanceof ApiError && e.status === 422) {
            shadowCache.current.set(frame, { unavailable: e.detail });
            setSh({ status: "success", data: null, error: null, unavailable: e.detail });
            return;
          }
          setSh((s) => ({ ...s, status: "error", error: e instanceof Error ? e : new Error(String(e)), unavailable: null }));
        },
      );
    }, SHADOW_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [shadowWanted, frame, replay.matchId, shNonce]);
  const shadow = sh.data && sh.data.frame === frame ? sh.data : null;

  // ---- scenario ----
  // Defender response policies: read from the backend when it offers the list (lazily, in edit mode),
  // otherwise the two policies every backend version accepts.
  const pol = useAsync<PolicyInfo[]>(mode === "edit" ? `policies:${replay.matchId}` : null, () =>
    loadPolicies(replay.matchId),
  );
  const policies = useMemo(
    () => orderPolicies(pol.data && pol.data.length > 0 ? pol.data : FALLBACK_POLICIES),
    [pol.data],
  );
  const scCache = useRef(new Map<string, ScenarioResponse>());
  // Each result remembers the request it answers, so a result for the previous policy or value
  // model is never presented under the new choice while the new request is in flight.
  const scKey = useMemo(
    () =>
      mode === "edit" && interventions.length > 0
        ? JSON.stringify(scenarioRequest(replay.matchId, frame, interventions, policy, valueModel))
        : null,
    [mode, interventions, policy, valueModel, frame, replay.matchId],
  );
  // The same edit with static defenders, so the panel can show how much the result depends on the
  // response policy. Only needed when a non-static policy is chosen; usually already cached.
  const stKey = useMemo(
    () =>
      mode === "edit" && interventions.length > 0 && policy !== STATIC_POLICY
        ? JSON.stringify(scenarioRequest(replay.matchId, frame, interventions, STATIC_POLICY, valueModel))
        : null,
    [mode, interventions, policy, valueModel, frame, replay.matchId],
  );
  const [sc, setSc] = useState<KeyedLoadable<ScenarioResponse>>(KEYED_IDLE);
  const [st, setSt] = useState<KeyedLoadable<ScenarioResponse>>(KEYED_IDLE);
  const [scNonce, setScNonce] = useState(0);
  useScenarioRequest(scKey, scCache.current, setSc, scNonce, true);
  useScenarioRequest(stKey, scCache.current, setSt, scNonce, false);
  const scStatus = sc.status === "success" && sc.key !== scKey ? "loading" : sc.status;
  const scenario = scKey !== null ? sc.data : null;
  const staticScenario = stKey !== null && st.key === stKey ? st.data : null;
  const stStatus = st.key === stKey ? st.status : "loading";

  const pending = useMemo(() => {
    if (mode !== "edit" || interventions.length === 0 || scStatus === "success") return NO_PENDING;
    return new Map(interventions.map((i) => [i.player_id, { x: i.x, y: i.y }]));
  }, [mode, interventions, scStatus]);

  // ---- push React-owned scene state to the renderer ----
  useEffect(() => {
    rendererRef.current?.setScene({
      replay,
      playing,
      analysis,
      scenario,
      view,
      layers,
      editMode: mode === "edit",
      pending,
      selectedId: selected,
      highlightReceiver: highlight,
      tracePlayers: trace,
      teams,
      shadow,
    });
  }, [rendererReady, replay, playing, analysis, scenario, view, layers, mode, pending, selected, highlight, trace, teams, shadow]);

  // A traced statement belongs to one frame and one mode; drop the rings when either changes.
  useEffect(() => setTrace(NO_TRACE), [frame, mode, playing]);
  const onTrace = useCallback((ids: string[]) => setTrace(ids.length ? ids : NO_TRACE), []);

  const onReady = useCallback(
    (r: PitchRenderer | null) => {
      rendererRef.current = r;
      if (r) {
        r.setScene({ t: playback.t }, false);
        setRendererReady((n) => n + 1);
      }
    },
    [playback],
  );

  // ---- mode transitions ----
  const enterEdit = useCallback(() => {
    playback.pause();
    setMode("edit");
    setView("hypothetical");
    setNotice(null);
  }, [playback]);
  const resetScenario = useCallback(() => {
    setInterventions([]);
    setSc(KEYED_IDLE);
    setView("hypothetical");
    setNotice(null);
  }, []);
  const exitEdit = useCallback(() => {
    setMode("analyse");
    setInterventions([]);
    setSc(KEYED_IDLE);
    setView("observed");
    setNotice(null);
  }, []);

  // ---- editing ----
  const observedAt = useCallback(
    (id: string): { x: number; y: number } | null => {
      const ps = analysis?.players.find((p) => p.player_id === id);
      if (ps && ps.x !== null && ps.y !== null) return { x: ps.x, y: ps.y };
      const k = replay.players.findIndex((p) => p.player_id === id);
      return k >= 0 ? valueAt(replay.xy, replay.nPlayers, index, k) : null;
    },
    [analysis, replay, index],
  );

  const applyMove = useCallback(
    (id: string, x: number, y: number) => {
      const obs = observedAt(id);
      const rx = Math.round(x * 100) / 100;
      const ry = Math.round(y * 100) / 100;
      setInterventions((list) => {
        const rest = list.filter((i) => i.player_id !== id);
        if (obs && Math.hypot(obs.x - x, obs.y - y) < 0.25) return rest;
        if (rest.length >= MAX_INTERVENTIONS) return list;
        return [...rest, { player_id: id, x: rx, y: ry }];
      });
      setView((v) => (v === "observed" ? "hypothetical" : v));
      setSelected(id);
    },
    [observedAt],
  );

  const onDragStart = useCallback(
    (p: DrawnPlayer): DragInit | null => {
      if (mode !== "edit") return null;
      if (!analysis) {
        setNotice("Wait for the observed analysis of this frame to finish loading, then drag.");
        return null;
      }
      if (!interventions.some((i) => i.player_id === p.id) && interventions.length >= MAX_INTERVENTIONS) {
        setNotice(`Up to ${MAX_INTERVENTIONS} players can be moved in one scenario. Undo a move first.`);
        return null;
      }
      const ps = analysis.players.find((q) => q.player_id === p.id);
      if (!ps || ps.x === null || ps.y === null) return null;
      setNotice(null);
      setView((v) => (v === "observed" ? "hypothetical" : v));
      return {
        fromX: ps.x,
        fromY: ps.y,
        originX: ps.reach_origin_x,
        originY: ps.reach_origin_y,
        radius: ps.reach_radius_m,
      };
    },
    [mode, analysis, interventions],
  );

  const onDrop = useCallback(
    (id: string, x: number, y: number) => {
      const ps = analysis?.players.find((q) => q.player_id === id);
      if (ps && ps.reach_origin_x !== null && ps.reach_origin_y !== null && ps.reach_radius_m !== null) {
        const out = Math.hypot(x - ps.reach_origin_x, y - ps.reach_origin_y) > ps.reach_radius_m;
        setNotice(
          out
            ? `${playerLabel(id, players)} was dropped outside the reachable region, so the backend will flag this edit as not supported.`
            : null,
        );
      }
      applyMove(id, x, y);
    },
    [analysis, applyMove, players],
  );

  const onDragOutsideChange = useCallback((outside: boolean | null, id: string | null) => {
    setDragWarn(id ? { outside, id } : null);
  }, []);

  const retryAnalysis = useCallback(() => setAnNonce((n) => n + 1), []);
  const retryShadow = useCallback(() => setShNonce((n) => n + 1), []);
  const retryScenario = useCallback(() => setScNonce((n) => n + 1), []);
  const jumpTo = useCallback((i: number) => playback.step(i - Math.round(playback.t)), [playback]);
  const closeInspector = useCallback(() => setSelected(null), []);
  const removeMove = useCallback((id: string) => setInterventions((l) => l.filter((i) => i.player_id !== id)), []);

  // Player menu: keyboard and touch users choose a player here instead of on the canvas.
  const present = useMemo(() => {
    const out: PlayerOut[] = [];
    replay.players.forEach((pl, k) => {
      if (Number.isFinite(replay.xy[(index * replay.nPlayers + k) * 2] as number)) out.push(pl);
    });
    return out.sort((a, b) => (a.team === b.team ? a.jersey - b.jersey : a.team === "home" ? -1 : 1));
  }, [replay, index]);
  const pickFromMenu = useCallback(
    (id: string) => {
      setSelected(id || null);
      if (id && mode === "edit") setFocusMove((n) => n + 1);
    },
    [mode],
  );

  // ---- keyboard shortcuts ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable) return;
      if (e.key === " " || e.code === "Space") {
        if (tag === "BUTTON" || tag === "SUMMARY" || tag === "A") return;
        if (mode === "edit") return;
        e.preventDefault();
        playback.toggle();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        if (el?.closest('[role="radiogroup"]')) return;
        if (mode === "edit") return;
        e.preventDefault();
        const n = e.shiftKey ? Math.round(replay.fps / replay.step) : 1;
        playback.step(e.key === "ArrowLeft" ? -n : n);
      } else if (e.key === "Escape") {
        setSelected(null);
        setHighlight(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playback, mode, replay.fps, replay.step]);

  // ---- derived display data ----
  const clock = clockAt(replay, index, match);
  const period = periodAt(replay, index);
  const shownAnalysis = scenario ? (view === "observed" ? scenario.observed : scenario.hypothetical) : analysis;
  const poss = replay.possession;

  const inspected: InspectedPlayer | null = useMemo(() => {
    if (!selected) return null;
    const pl = players.get(selected);
    if (!pl) return null;
    const src = scenario && view !== "observed" ? scenario.hypothetical : analysis;
    const ps = src?.players.find((q) => q.player_id === selected);
    if (ps) {
      const pend = pending.get(selected);
      return {
        player: pl,
        x: pend ? pend.x : ps.x,
        y: pend ? pend.y : ps.y,
        speed: speedOf(ps.vx, ps.vy),
        speedSource: "(from the analysis velocity)",
        positionProv: pend ? "hypothetical" : ps.position_provenance,
        velocityProv: ps.velocity_provenance,
        isGk: ps.is_gk,
      };
    }
    const k = replay.players.findIndex((q) => q.player_id === selected);
    const v = k >= 0 ? valueAt(replay.xy, replay.nPlayers, index, k) : null;
    return {
      player: pl,
      x: v?.x ?? null,
      y: v?.y ?? null,
      speed: k >= 0 ? finiteDiffSpeed(replay.xy, replay.nPlayers, replay.nFrames, index, k, replay.step / replay.fps) : null,
      speedSource: "(finite difference of tracking)",
      // broadcast tracking: off-camera positions are provider extrapolations
      positionProv: k >= 0 && replay.detected && replay.detected[index * replay.nPlayers + k] === 0 ? "estimated" : "observed",
      velocityProv: "derived",
      isGk: null,
    };
  }, [selected, players, scenario, view, analysis, pending, replay, index]);

  const canvasLabel = useMemo(() => {
    const parts = [`Tactical pitch, period ${period}, ${clock}.`];
    if (playing) parts.push("Replay playing.");
    else if (shownAnalysis) {
      parts.push(
        shownAnalysis.attacking_team
          ? `${teamName(shownAnalysis.attacking_team, teams)} in possession${
              shownAnalysis.carrier_id ? `, carrier ${playerLabel(shownAnalysis.carrier_id, players)}` : ""
            }.`
          : "No team in possession.",
      );
      parts.push(`${shownAnalysis.pass_options.length} pass options, listed in the pass options table.`);
    }
    if (replay.detected) {
      let shown = 0;
      let est = 0;
      for (let k = 0; k < replay.nPlayers; k++) {
        if (!Number.isFinite(replay.xy[(index * replay.nPlayers + k) * 2] as number)) continue;
        shown++;
        if (replay.detected[index * replay.nPlayers + k] === 0) est++;
      }
      parts.push(`${est} of ${shown} player positions are provider estimates for players outside the broadcast camera view.`);
    }
    if (scenario) parts.push(`Showing the ${view} view of an edited scenario.`);
    return parts.join(" ");
  }, [period, clock, playing, shownAnalysis, players, teams, scenario, view, replay, index]);

  const pitchStatus =
    mode === "edit"
      ? scStatus === "loading"
        ? "Re-running model..."
        : scenario
          ? view === "observed"
            ? "Observed frame"
            : view === "hypothetical"
              ? "Hypothetical frame"
              : "Difference: hypothetical − observed"
          : "Edit scenario: drag a player"
      : playing
        ? null
        : an.status === "loading"
          ? "Analysing frame..."
          : analysis
            ? analysis.pitch_control || analysis.pass_options.length > 0
              ? "Paused · model overlay"
              : "Paused · no model overlay at this frame"
            : null;

  return (
    <div className="lab" data-mode={mode}>
      <div className="lab-head">
        <BackButton onBack={onBack} />
        <div className="lab-title">
          {match ? (
            <p className="lab-match small" data-testid="match-title">
              {matchTitle(match)}
            </p>
          ) : null}
          <h1 className="h-page">
            {poss ? (
              <>
                <span className={`team-chip team-chip--${poss.team}`}>{teamName(poss.team, teams)}</span> possession · P
                {poss.period} {clockAt(replay, indexOfFrame(replay.frame, poss.start_frame), match)}
              </>
            ) : (
              "Replay"
            )}
          </h1>
          {poss ? (
            <p className="muted small">
              {poss.duration_s.toFixed(1)} s · {poss.n_passes} {poss.n_passes === 1 ? "pass" : "passes"} · {titleCase(poss.first_type)} →{" "}
              {titleCase(poss.last_type)}
              {poss.last_subtype ? ` (${titleCase(poss.last_subtype)})` : ""}
              {poss.has_goal ? <span className="badge badge--goal">Goal</span> : poss.has_shot ? <span className="badge badge--shot">Shot</span> : null}
            </p>
          ) : null}
        </div>
        <Segmented<Mode>
          label="Mode"
          hideLabel
          value={mode}
          onChange={(m) => (m === "edit" ? enterEdit() : exitEdit())}
          testId="mode"
          options={[
            { value: "analyse", label: "Replay & analyse" },
            { value: "edit", label: "Edit scenario" },
          ]}
        />
      </div>

      <div className="lab-grid">
        <section className="stage" aria-label="Pitch">
          <div className="stage-toolbar">
            <fieldset className="layer-toggles">
              <legend className="sr-only">Overlay layers</legend>
              {(
                [
                  ["control", "Pitch control"],
                  ["passes", "Pass options"],
                  ["pressure", "Pressure"],
                  ["trails", "Trails"],
                ] as [keyof Layers, string][]
              ).map(([k, label]) => (
                <label key={k} className="chip-toggle">
                  <input
                    type="checkbox"
                    checked={layers[k]}
                    onChange={(e) => setLayers((l) => ({ ...l, [k]: e.target.checked }))}
                    data-testid={`layer-${k}`}
                  />
                  <span>{label}</span>
                </label>
              ))}
              <label
                className={`chip-toggle chip-toggle--experimental${shadowAvailable ? "" : " is-disabled"}`}
                title={
                  shadowAvailable
                    ? "Experimental: estimated passing value each defender's position removes (about 0.4 s per frame)"
                    : "Available when paused in Replay & analyse mode"
                }
              >
                <input
                  type="checkbox"
                  checked={layers.shadow}
                  disabled={!shadowAvailable}
                  onChange={(e) => setLayers((l) => ({ ...l, shadow: e.target.checked }))}
                  data-testid="layer-shadow"
                />
                <span>Defensive shadow (experimental)</span>
              </label>
            </fieldset>
            <label className="player-menu">
              <span className="segmented-label">Player</span>
              <select
                className="select select--sm"
                value={selected ?? ""}
                onChange={(e) => pickFromMenu(e.target.value)}
                data-testid="player-menu"
                title={mode === "edit" ? "Choose a player to move" : "Choose a player to inspect"}
              >
                <option value="">{mode === "edit" ? "Choose a player to move" : "Choose a player"}</option>
                {(["home", "away"] as const).map((team) => (
                  <optgroup key={team} label={teamName(team, teams)}>
                    {present
                      .filter((pl) => pl.team === team)
                      .map((pl) => (
                        <option key={pl.player_id} value={pl.player_id}>
                          #{pl.jersey} {teamName(pl.team, teams)}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </label>
            {mode === "edit" ? (
              <Segmented<ScenarioView>
                label="Scenario view"
                hideLabel
                value={view}
                onChange={setView}
                testId="scenario-view"
                options={[
                  { value: "observed", label: "Observed", title: "The frame as tracked" },
                  { value: "hypothetical", label: "Hypothetical", title: "Your edited frame" },
                  { value: "difference", label: "Difference", title: "Estimated control: hypothetical minus observed" },
                ]}
              />
            ) : null}
          </div>

          <div className="pitch-frame">
            <PitchCanvas
              replay={replay}
              editable={mode === "edit"}
              ariaLabel={canvasLabel}
              describedBy={PASS_TABLE_ID}
              onReady={onReady}
              onPick={setSelected}
              onDragStart={onDragStart}
              onDragOutsideChange={onDragOutsideChange}
              onDrop={onDrop}
            />
            {pitchStatus ? (
              <span className={`pitch-status${mode === "edit" ? " pitch-status--edit" : ""}`} data-testid="pitch-status">
                {pitchStatus}
              </span>
            ) : null}
            {dragWarn ? (
              <span
                className={`drag-warn ${dragWarn.outside ? "drag-warn--bad" : dragWarn.outside === false ? "drag-warn--ok" : ""}`}
                role="status"
                data-testid="drag-warning"
              >
                {dragWarn.outside === true
                  ? "Outside the reachable region: not reachable at 7 m/s from the observed position 1.5 s earlier"
                  : dragWarn.outside === false
                    ? "Within the reachable region"
                    : "Reachability cannot be verified: no observed position 1.5 s earlier"}
              </span>
            ) : null}
          </div>

          <Transport
            playback={playback}
            replay={replay}
            match={match}
            playing={playing}
            speed={speed}
            onSpeed={setSpeed}
            disabled={mode === "edit"}
            disabledReason="Playback is paused while you edit a scenario. Switch back to Replay & analyse to resume."
          />
          {mode === "edit" ? (
            <p className="small muted transport-note">
              Playback is locked to frame {frame} while you edit. Switch back to <em>Replay &amp; analyse</em> to resume.
            </p>
          ) : (
            <p className="small muted transport-note transport-note--keys">
              <kbd>Space</kbd> play/pause · <kbd>←</kbd> <kbd>→</kbd> step one frame (<kbd>Shift</kbd> for 1 s) · click a
              player to inspect
            </p>
          )}

          {scenario && mode === "edit" ? <ScenarioJump scenario={scenario} players={players} /> : null}

          <Legend
            attackingTeam={shownAnalysis?.attacking_team ?? poss?.team ?? null}
            showHeat={layers.control && !playing}
            showEstimated={replay.nEstimated > 0}
            showShadow={layers.shadow && !!shadow && shadowAvailable}
            scenarioLayer={mode === "edit"}
            view={scenario ? view : "observed"}
            teams={teams}
          />
        </section>

        <aside className="side" aria-label="Analysis">
          {inspected ? (
            <PlayerInspector
              info={inspected}
              playing={playing}
              editable={mode === "edit"}
              onMove={(x, y) => selected && applyMove(selected, x, y)}
              onClose={closeInspector}
              focusMove={focusMove}
              teams={teams}
            />
          ) : null}

          {mode === "edit" ? (
            <ScenarioPanel
              ready={!!analysis}
              frameLabel={`P${period} ${clock}`}
              interventions={interventions}
              policy={policy}
              onPolicy={setPolicy}
              policies={policies}
              staticStatus={stStatus}
              staticScenario={staticScenario}
              valueModel={valueModel}
              onValueModel={setValueModel}
              status={scStatus}
              scenario={scenario}
              error={sc.error}
              onRetry={retryScenario}
              onRemove={removeMove}
              onReset={resetScenario}
              players={players}
              highlight={highlight}
              onHighlight={setHighlight}
              notice={notice}
            />
          ) : (
            <>
              {layers.shadow ? (
                <ShadowPanel
                  playing={playing}
                  status={sh.status}
                  shadow={shadow}
                  unavailable={sh.unavailable}
                  error={sh.error}
                  onRetry={retryShadow}
                  players={players}
                />
              ) : null}
              <WhyPanel
                analysis={an.data}
                frame={frame}
                playing={playing}
                loading={an.status === "loading"}
                players={players}
                onTrace={onTrace}
              />
              <AnalysisPanel
                teams={teams}
                playing={playing}
                status={an.status}
                analysis={an.data}
                error={an.error}
                onRetry={retryAnalysis}
                players={players}
                highlight={highlight}
                onHighlight={setHighlight}
                clock={`P${period} ${clock}`}
              />
              <SimilarPanel
                matchId={replay.matchId}
                frame={playing ? null : frame}
                provider={providerInfo(replay.source).label}
                players={players}
              />
              <EventsCard replay={replay} match={match} players={players} onJump={jumpTo} />
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

/**
 * Stacked layouts (tablet, phone) put the scenario panel below the pitch, out of sight after a
 * drag. This one-line result links down to it. Hidden on wide screens, where the panel is beside
 * the pitch.
 */
function ScenarioJump({ scenario, players }: { scenario: ScenarioResponse; players: Map<string, PlayerOut> }) {
  const top = [...scenario.option_deltas].sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0))[0];
  const go = () => {
    const el = document.getElementById("sc-title");
    el?.scrollIntoView({ block: "start" });
    el?.focus({ preventScroll: true });
  };
  return (
    <button type="button" className="scenario-jump" onClick={go} data-testid="scenario-jump">
      <span>
        {top && !isNegligible(top.delta) ? (
          <>
            Largest change: pass to <strong>{playerLabel(top.receiver_id, players)}</strong>{" "}
            <span className={(top.delta ?? 0) > 0 ? "pos" : "neg"}>{fmtSigned(top.delta)}</span>
          </>
        ) : (
          "No pass option changes noticeably"
        )}
        {scenario.supported ? "" : " · edit not supported"}
      </span>
      <span className="scenario-jump-cta">
        Full comparison <span aria-hidden="true">↓</span>
      </span>
    </button>
  );
}

const EventsCard = memo(function EventsCard({
  replay,
  match,
  players,
  onJump,
}: {
  replay: ReplayData;
  match: MatchSummary | null;
  players: Map<string, PlayerOut>;
  onJump: (idx: number) => void;
}) {
  if (replay.events.length === 0) return null;
  return (
    <section className="card" aria-labelledby="ev-list-title">
      <h2 id="ev-list-title" className="h-card">
        Events <span className="prov prov--observed">observed</span>
      </h2>
      <ol className="events">
        {replay.events.map((e) => {
          const idx = indexOfFrame(replay.frame, e.start_frame);
          return (
            <li key={e.event_id}>
              <button type="button" className="event-btn" onClick={() => onJump(idx)}>
                <span className="tabular muted">{clockAt(replay, idx, match)}</span>
                <span className={`event-dot event-dot--${e.team}`} aria-hidden="true" />
                <span>
                  {eventLabel(e)} · {playerLabel(e.from_player_id, players)}
                  {e.to_player_id ? ` → ${playerLabel(e.to_player_id, players)}` : ""}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
});
