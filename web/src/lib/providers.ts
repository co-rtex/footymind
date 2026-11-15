/**
 * Per-provider wording for grouping, attribution and the provenance caveat.
 * Keyed by SourceInfo.provider; unknown providers fall back to generic text.
 */
import type { MatchSummary, SourceInfo } from "../api/types";

export interface ProviderInfo {
  id: string;
  label: string;
  /** one line describing what kind of tracking this is */
  kind: string;
  /** phrase used in the attribution: "<source.name>: <attribution>" */
  attribution: string;
  anonymised: boolean;
}

const KNOWN: Record<string, Omit<ProviderInfo, "id">> = {
  metrica: {
    label: "Metrica Sports",
    kind: "Optical tracking of every player at 25 Hz with synchronised events. Every drawn position is observed.",
    attribution: "real, anonymised tracking and synchronised event data published by Metrica Sports",
    anonymised: true,
  },
  skillcorner: {
    label: "SkillCorner",
    kind:
      "Broadcast tracking at 10 Hz. Players outside the camera view have positions extrapolated by SkillCorner; the lab draws those as estimates (hollow, dashed outline), not observations.",
    attribution: "broadcast tracking data published by SkillCorner (credit: SkillCorner)",
    anonymised: false,
  },
};

export function providerInfo(source: Pick<SourceInfo, "provider">): ProviderInfo {
  const k = KNOWN[source.provider];
  if (k) return { id: source.provider, ...k };
  return {
    id: source.provider,
    label: source.provider,
    kind: "Tracking data.",
    attribution: `tracking data published by ${source.provider}`,
    anonymised: false,
  };
}

export interface ProviderGroup {
  info: ProviderInfo;
  /** source URL of the first match in the group (credited in the group header) */
  url: string;
  matches: MatchSummary[];
}

/**
 * Split a group's source names into a shared prefix (shown once) and per-match
 * labels, cutting on word boundaries and never leaving a bare number:
 * "SkillCorner open data, A-League 2024/25 match 1874553" gives the prefix
 * "SkillCorner open data, A-League 2024/25" and the label "Match 1874553".
 * Returns null when the names share no useful prefix.
 */
export function splitCommonName(names: string[]): { prefix: string; labels: string[] } | null {
  if (names.length < 2) return null;
  let lcp = names[0] ?? "";
  for (const n of names) {
    let i = 0;
    while (i < lcp.length && i < n.length && lcp[i] === n[i]) i++;
    lcp = lcp.slice(0, i);
  }
  let cut = lcp.lastIndexOf(" ") + 1;
  // step back a word while any remainder would start with a digit (e.g. "1874553")
  while (cut > 0 && names.some((n) => /^\d/.test(n.slice(cut)))) {
    cut = lcp.lastIndexOf(" ", cut - 2) + 1;
  }
  if (cut <= 0) return null;
  const prefix = lcp.slice(0, cut).replace(/[\s,;:·-]+$/, "");
  const labels = names.map((n) => {
    const r = n.slice(cut).trim();
    return r.charAt(0).toUpperCase() + r.slice(1);
  });
  if (!prefix || labels.some((l) => !l)) return null;
  return { prefix, labels };
}

/** Group matches by provider, keeping the API's order of first appearance. */
export function groupByProvider(matches: MatchSummary[]): ProviderGroup[] {
  const groups = new Map<string, ProviderGroup>();
  for (const m of matches) {
    const g = groups.get(m.source.provider);
    if (g) g.matches.push(m);
    else groups.set(m.source.provider, { info: providerInfo(m.source), url: m.source.url, matches: [m] });
  }
  return [...groups.values()];
}
