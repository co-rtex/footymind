/**
 * Live numbers for the featured situations on the landing page.
 *
 * Requests are de-duplicated (one in-flight or settled promise per situation, so remounts and
 * React's development double-effects do not recompute on the server) and run one at a time:
 * both scenarios first, so the main numbers appear quickly, then the slower Defensive Shadow
 * calls. A failed request is forgotten, so Retry asks again.
 */
import { api } from "../api/client";
import type { DefensiveShadowOut, ScenarioResponse } from "../api/types";
import type { FeaturedScenario } from "./featured";
import { scenarioRequest, SHADOW_OPTS } from "./scenario";

let queue: Promise<unknown> = Promise.resolve();
const cache = new Map<string, Promise<unknown>>();

function serial<T>(fn: () => Promise<T>): Promise<T> {
  const p = queue.then(fn, fn);
  queue = p.catch(() => undefined);
  return p;
}

function cached<T>(key: string, fn: () => Promise<T>): Promise<T> {
  let p = cache.get(key) as Promise<T> | undefined;
  if (!p) {
    p = serial(fn);
    cache.set(key, p);
    p.catch(() => cache.delete(key));
  }
  return p;
}

/** The featured move with static defenders: exactly the request the lab makes for it. */
export function featuredScenario(f: FeaturedScenario): Promise<ScenarioResponse> {
  return cached(`scenario:${f.id}`, () =>
    api.scenario(scenarioRequest(f.matchId, f.frame, f.moves, "none", "option_model")),
  );
}

/** The Defensive Shadow at the featured frame, requested after the scenario; null if unavailable. */
export function featuredShadow(f: FeaturedScenario): Promise<DefensiveShadowOut | null> {
  return featuredScenario(f)
    .catch(() => null)
    .then(() => cached(`shadow:${f.id}`, () => api.defensiveShadow(f.matchId, f.frame, SHADOW_OPTS)))
    .catch(() => null);
}

/** Tests only: forget cached results. */
export function resetFeaturedLive(): void {
  cache.clear();
  queue = Promise.resolve();
}
