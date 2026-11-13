/**
 * Visual encoding of player-position provenance on the pitch.
 *
 *   observed      solid team-colour disc, dark number
 *   estimated     hollow disc: faint team fill, dashed TEAM-colour ring, team-colour number
 *                 (provider extrapolation of an off-camera player)
 *   hypothetical  translucent disc, dashed VIOLET (accent) ring (moved by the user)
 *   simulated     translucent disc, dotted WHITE ring (moved by a response rule)
 *
 * Estimated and hypothetical are both dashed, so they differ in ring colour
 * (team vs violet), fill (near-empty vs translucent) and dash rhythm.
 */
import type { ProvenanceLabel } from "../api/types";

export type PlayerStyleKind = "observed" | "estimated" | "hypothetical" | "simulated";

export function styleKindFor(prov: ProvenanceLabel): PlayerStyleKind {
  switch (prov) {
    case "estimated":
    case "hypothetical":
    case "simulated":
      return prov;
    default:
      // "derived" never applies to positions; treat it like observed rather than inventing a style
      return "observed";
  }
}

export interface DiscStyle {
  /** opacity of the team-colour fill (1 = solid) */
  fillAlpha: number;
  /** opacity of a dark pitch-colour backing under the fill, so hollow discs read on any overlay */
  backingAlpha: number;
  ring: "outline" | "dashed" | "dotted";
  ringColour: "shadow" | "team" | "accent" | "text";
  ringAlpha: number;
  ringWidth: number;
  dash: number[];
  numberColour: "onTeam" | "team" | "text";
  numberAlpha: number;
}

export function discStyle(kind: PlayerStyleKind): DiscStyle {
  switch (kind) {
    case "estimated":
      return {
        fillAlpha: 0.14,
        backingAlpha: 0.8,
        ring: "dashed",
        ringColour: "team",
        ringAlpha: 1,
        ringWidth: 2,
        dash: [3, 2.2],
        numberColour: "team",
        numberAlpha: 0.95,
      };
    case "hypothetical":
      return {
        fillAlpha: 0.3,
        backingAlpha: 0,
        ring: "dashed",
        ringColour: "accent",
        ringAlpha: 1,
        ringWidth: 2,
        dash: [4, 2.5],
        numberColour: "text",
        numberAlpha: 1,
      };
    case "simulated":
      return {
        fillAlpha: 0.3,
        backingAlpha: 0,
        ring: "dotted",
        ringColour: "text",
        ringAlpha: 1,
        ringWidth: 2,
        dash: [1.2, 2.4],
        numberColour: "text",
        numberAlpha: 1,
      };
    default:
      return {
        fillAlpha: 1,
        backingAlpha: 0,
        ring: "outline",
        ringColour: "shadow",
        ringAlpha: 0.45,
        ringWidth: 1,
        dash: [],
        numberColour: "onTeam",
        numberAlpha: 1,
      };
  }
}

/** Trail segments that touch an extrapolated frame are drawn dashed and faded. */
export const ESTIMATED_TRAIL = { dash: [2, 3], alphaScale: 0.55 } as const;
