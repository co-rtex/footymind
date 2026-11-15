/**
 * Live numbers for the featured situations on the landing page.
 *
 * Requests are de-duplicated (one in-flight or settled promise per situation, so remounts and
 * React's development double-effects do not recompute on the server) and run one at a time:
 * the scenarios first (static defenders, and the defender response where a situation has one),
 * so the main numbers appear quickly, then the slower Defensive Shadow calls. A failed request
 * is forgotten, so Retry asks again.
 */
import { api } from "../api/client";
import type { DefensiveShadowOut, ScenarioResponse } from "../api/types";
import type { FeaturedScenario } from "./featured";
import { STATIC_POLICY } from "./policies";
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
    api.scenario(scenarioRequest(f.matchId, f.frame, f.moves, STATIC_POLICY, "option_model")),
  );
}

/** The same move under the situation's defender response (the lab's request for it); null when it has none. */
export function featuredResponse(f: FeaturedScenario): Promise<ScenarioResponse | null> {
  const policy = f.response;
  if (!policy || policy === STATIC_POLICY) return Promise.resolve(null);
  return cached(`response:${f.id}`, () =>
    api.scenario(scenarioRequest(f.matchId, f.frame, f.moves, policy, "option_model")),
  );
}

/** Every scenario a featured card needs; fails if either fails. */
export function featuredScenarios(
  f: FeaturedScenario,
): Promise<{ scenario: ScenarioResponse; response: ScenarioResponse | null }> {
  return Promise.all([featuredScenario(f), featuredResponse(f)]).then(([scenario, response]) => ({ scenario, response }));
}

/**
 * The Defensive Shadow at the featured frame, requested after the scenario; null if unavailable
 * or if the situation does not show it (never requested then).
 */
export function featuredShadow(f: FeaturedScenario): Promise<DefensiveShadowOut | null> {
  if (!f.showShadow) return Promise.resolve(null);
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
