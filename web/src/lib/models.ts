/**
 * Which per-option probability to show, and how to label it.
 *
 * Primary: the data-calibrated option model (`p_success_model`).
 * Secondary: the physics heuristic (`p_success`, uncalibrated).
 * If the option model is unavailable for a frame we fall back to physics and
 * say so, never silently mixing the two.
 */
import type { ModelInfo, PassOptionOut, ValueModel } from "../api/types";

export function primaryP(o: PassOptionOut): number {
  return o.p_success_model ?? o.p_success;
}

export function hasOptionModel(options: PassOptionOut[]): boolean {
  return options.length > 0 && options.every((o) => o.p_success_model !== null);
}

/** Number of real passes the option model was trained on, parsed from its status string. */
export function trainingPasses(info: ModelInfo | null | undefined): number | null {
  const m = info?.status.match(/trained on ([\d,]+) (?:attempted |real )?passes/i);
  if (!m || !m[1]) return null;
  const n = Number(m[1].replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

export function optionModelLabel(info: ModelInfo | null | undefined): string {
  const n = trainingPasses(info);
  return n !== null ? `Option model (calibrated on ${n.toLocaleString("en-GB")} real passes)` : "Option model (calibrated on real passes)";
}

export const PHYSICS_LABEL = "Physics heuristic (uncalibrated)";

export function valueModelLabel(v: ValueModel, info?: ModelInfo | null): string {
  return v === "option_model" ? optionModelLabel(info) : PHYSICS_LABEL;
}
