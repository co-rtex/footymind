/**
 * Display rules for the experimental Defensive Shadow layer.
 *
 * The halo is scaled on an ABSOLUTE scale (not relative to the frame's
 * maximum), so a large halo always means a large estimated suppression and
 * halos are comparable across frames. Defenders below the noise floor, or with
 * a non-positive shadow, get no halo.
 */
import type { DefenderShadowOut } from "../api/types";

/** "a 3.5 m ring" when the API has sent the radius (alt_radius_m), else "a ring". Never a constant of the app. */
export function ringText(altRadiusM: number | null | undefined): string {
  return typeof altRadiusM === "number" && Number.isFinite(altRadiusM) && altRadiusM > 0 ? `a ${altRadiusM} m ring` : "a ring";
}

/**
 * The panel's caveat, with the ring radius the API sent (DefensiveShadowOut.alt_radius_m) once it has arrived.
 * Once the API has answered for a frame, the validation sentence is the API's own status for that match's
 * provider (DefensiveShadowOut.status, with its ledger claim); before that, the summary for both providers.
 */
export function shadowCaveat(
  altRadiusM: number | null | undefined,
  served?: { status?: string | null; validation_claim?: string | null } | null,
): string {
  const status = served?.status?.trim();
  if (status) {
    const id = served?.validation_claim;
    const claim = id && !status.includes(id) ? ` (claims ledger ${id})` : "";
    return `Experimental model-based sensitivity: how much passing value (option model) this position removes compared with nearby positions on ${ringText(altRadiusM)} around the defender (not checked for reachability). Validation for this match's provider${claim}: ${status}`;
  }
  return `Experimental model-based sensitivity: how much passing value (option model) this position removes compared with nearby positions on ${ringText(altRadiusM)} around the defender (not checked for reachability). Validated only partly: on Metrica data it ranks the eventual interceptor first about 2× more often than chance, but not better than simply 'nearest defender to the ball'; on SkillCorner broadcast data it does no better than chance. Not a defender rating.`;
}

/** shadow_sum at which the halo reaches full size and opacity (summed option value). */
export const SHADOW_FULL_SCALE = 0.3;
export const HALO_MIN_RADIUS_M = 1.6;
export const HALO_MAX_RADIUS_M = 5;

/** Only distinguishable, suppressive positions are drawn. */
export function shadowDrawn(d: DefenderShadowOut): boolean {
  return d.distinguishable_from_noise && d.shadow_sum > 0;
}

/** Normalised strength in [0, 1], square-root scaled so that halo AREA tracks shadow_sum. */
export function shadowStrength(shadowSum: number, fullScale = SHADOW_FULL_SCALE): number {
  if (!Number.isFinite(shadowSum) || shadowSum <= 0 || fullScale <= 0) return 0;
  return Math.sqrt(Math.min(1, shadowSum / fullScale));
}

/** Halo radius (metres) and peak opacity for a defender, or null when no halo should be drawn. */
export function shadowHalo(d: DefenderShadowOut): { radiusM: number; alpha: number } | null {
  if (!shadowDrawn(d)) return null;
  const s = shadowStrength(d.shadow_sum);
  return {
    radiusM: HALO_MIN_RADIUS_M + (HALO_MAX_RADIUS_M - HALO_MIN_RADIUS_M) * s,
    alpha: 0.25 + 0.5 * s,
  };
}

/** Ranked for the side panel: most suppressive (largest shadow_sum) first. */
export function rankShadow(defenders: DefenderShadowOut[]): DefenderShadowOut[] {
  return [...defenders].sort((a, b) => b.shadow_sum - a.shadow_sum);
}

export type ShadowStanding = "suppressive" | "less-suppressive" | "within-noise";

/** How a defender's position compares with its reachable alternatives. */
export function shadowStanding(d: DefenderShadowOut): ShadowStanding {
  if (!d.distinguishable_from_noise) return "within-noise";
  return d.shadow_sum > 0 ? "suppressive" : "less-suppressive";
}

/** The suppressed receiver is only meaningful when the delta is positive. */
export function suppressedReceiver(d: DefenderShadowOut): string | null {
  return d.most_suppressed_receiver && d.most_suppressed_delta > 0 ? d.most_suppressed_receiver : null;
}
