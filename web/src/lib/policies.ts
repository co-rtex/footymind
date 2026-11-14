/**
 * Defender response policies for the Possession Multiverse.
 *
 * A policy decides how unmoved defenders react to the user's edit. "none" is the static
 * baseline (only the moved players change). Every other policy is a model of behaviour, so
 * the lab shows how much a conclusion depends on it by comparing against the static result.
 */
import type { OptionDeltaOut, PolicyInfo, ResponsePolicy } from "../api/types";
import type { PlayerOut } from "../api/types";
import { isNegligible } from "./agreement";
import { playerLabel } from "./format";

export const STATIC_POLICY: ResponsePolicy = "none";

/**
 * Used when the backend has no GET /api/response-policies (older versions). These two
 * policies have always been accepted by POST /api/scenarios.
 */
export const FALLBACK_POLICIES: PolicyInfo[] = [
  {
    name: "none",
    description: "Static: only the players you move change position; every other player stays where they were tracked.",
    assumptions: ["Unmoved players keep their observed positions and velocities."],
    validated: false,
    validation_ref: null,
  },
  {
    name: "marker_follows",
    description:
      "Rule: when you move an attacker, the nearest defender within 10 m shifts by half of the attacker's displacement.",
    assumptions: ["A simple hand-written rule, not fitted to how defenders actually move."],
    validated: false,
    validation_ref: null,
  },
];

const LABELS: Record<string, string> = {
  none: "Static (no response)",
  marker_follows: "Marker follows",
  press_carrier: "Press the carrier",
  zonal_shift: "Zonal shift",
  lane_cover: "Cover passing lanes",
  learned: "Learned from tracking",
};

/** Short human label for a policy name; unknown names are made readable, never hidden. */
export function policyLabel(name: string): string {
  const known = LABELS[name];
  if (known) return known;
  const words = name.replace(/[_-]+/g, " ").trim();
  return words ? words[0]!.toUpperCase() + words.slice(1) : name;
}

/** A behaviour model that has not been validated against observed defender movement. The static baseline is not a behaviour model. */
export function isExperimental(p: Pick<PolicyInfo, "name" | "validated"> | null | undefined): boolean {
  return !!p && p.name !== STATIC_POLICY && !p.validated;
}

/** Keep the static baseline first, then the backend's order. */
export function orderPolicies(list: PolicyInfo[]): PolicyInfo[] {
  const s = list.filter((p) => p.name === STATIC_POLICY);
  return [...s, ...list.filter((p) => p.name !== STATIC_POLICY)];
}

export interface PolicyDeltaRow {
  receiver_id: string;
  /** change in pass probability with static defenders */
  staticDelta: number | null;
  /** change in pass probability with the chosen response policy */
  policyDelta: number | null;
  /** policyDelta - staticDelta, null when either is missing */
  gap: number | null;
  /** the two changes point in opposite directions and neither is negligible */
  signFlip: boolean;
}

export interface PolicyComparison {
  rows: PolicyDeltaRow[];
  /** options whose direction of change differs between the two assumptions */
  nFlips: number;
  /** options present in both results */
  nCompared: number;
  /** largest |gap| across options */
  maxAbsGap: number | null;
  /** receiver with the largest |delta| under each assumption */
  topStatic: string | null;
  topPolicy: string | null;
}

function topOf(deltas: OptionDeltaOut[]): string | null {
  let best: OptionDeltaOut | null = null;
  for (const d of deltas) {
    if (d.delta === null) continue;
    if (!best || Math.abs(d.delta) > Math.abs(best.delta ?? 0)) best = d;
  }
  return best && !isNegligible(best.delta) ? best.receiver_id : null;
}

/** Join two scenario results (same edit, same value model) by receiver. */
export function comparePolicies(staticDeltas: OptionDeltaOut[], policyDeltas: OptionDeltaOut[]): PolicyComparison {
  const byId = new Map<string, PolicyDeltaRow>();
  const row = (id: string): PolicyDeltaRow => {
    let r = byId.get(id);
    if (!r) {
      r = { receiver_id: id, staticDelta: null, policyDelta: null, gap: null, signFlip: false };
      byId.set(id, r);
    }
    return r;
  };
  for (const d of staticDeltas) row(d.receiver_id).staticDelta = d.delta;
  for (const d of policyDeltas) row(d.receiver_id).policyDelta = d.delta;
  let nFlips = 0;
  let nCompared = 0;
  let maxAbsGap: number | null = null;
  for (const r of byId.values()) {
    if (r.staticDelta === null || r.policyDelta === null) continue;
    nCompared++;
    r.gap = r.policyDelta - r.staticDelta;
    if (maxAbsGap === null || Math.abs(r.gap) > maxAbsGap) maxAbsGap = Math.abs(r.gap);
    r.signFlip =
      !isNegligible(r.staticDelta) && !isNegligible(r.policyDelta) && Math.sign(r.staticDelta) !== Math.sign(r.policyDelta);
    if (r.signFlip) nFlips++;
  }
  const mag = (r: PolicyDeltaRow) => Math.max(Math.abs(r.staticDelta ?? 0), Math.abs(r.policyDelta ?? 0));
  const rows = [...byId.values()].sort((a, b) => mag(b) - mag(a));
  return { rows, nFlips, nCompared, maxAbsGap, topStatic: topOf(staticDeltas), topPolicy: topOf(policyDeltas) };
}

/** How much the conclusion depends on the opponent assumption, as a short verdict. */
export function sensitivityVerdict(c: PolicyComparison): "same" | "shifts" | "changes" {
  if (c.nFlips > 0 || c.topStatic !== c.topPolicy) return "changes";
  if (c.maxAbsGap !== null && c.maxAbsGap >= 0.05) return "shifts";
  return "same";
}

/** One plain-language sentence on whether the conclusion survives a change of opponent assumption. */
export function verdictText(c: PolicyComparison, policy: string, players: Map<string, PlayerOut>): string {
  const name = policyLabel(policy);
  const v = sensitivityVerdict(c);
  if (c.nCompared === 0) return "No pass options to compare at this frame.";
  if (v === "changes") {
    const parts: string[] = [];
    if (c.nFlips > 0) parts.push(`${c.nFlips} of ${c.nCompared} pass options change direction`);
    if (c.topStatic !== c.topPolicy) {
      parts.push(
        `the largest change moves from ${c.topStatic ? playerLabel(c.topStatic, players) : "no option"} (static) to ${
          c.topPolicy ? playerLabel(c.topPolicy, players) : "no option"
        } (${name})`,
      );
    }
    return `The conclusion depends on how defenders respond: ${parts.join(", and ")}.`;
  }
  if (v === "shifts") {
    return `Same direction under both assumptions, but the size of the changes differs by up to ${(c.maxAbsGap ?? 0).toFixed(2)}.`;
  }
  return `Little difference: with static defenders the changes have the same direction and differ by at most ${(c.maxAbsGap ?? 0).toFixed(2)}.`;
}
