/** Wording for ensemble sign agreement on a scenario delta. */

/** Below this absolute change the sign is noise, so agreement on it is not meaningful. */
export const NEGLIGIBLE_DELTA = 0.005;

export function isNegligible(delta: number | null): boolean {
  return delta === null || !Number.isFinite(delta) || Math.abs(delta) < NEGLIGIBLE_DELTA;
}

export function agreementLabel(a: number | null, delta?: number | null): { word: string; cls: string } {
  if (delta !== undefined && isNegligible(delta)) return { word: "no change", cls: "agree--na" };
  if (a === null || !Number.isFinite(a)) return { word: "n/a", cls: "agree--na" };
  if (a >= 0.9) return { word: "robust", cls: "agree--robust" };
  if (a >= 0.7) return { word: "mixed", cls: "agree--mostly" };
  return { word: "sensitive", cls: "agree--sensitive" };
}

export function agreementSentence(a: number | null, n: number, delta?: number | null): string {
  if (delta !== undefined && isNegligible(delta)) return "the change is negligible, so its direction is not meaningful";
  if (a === null || !Number.isFinite(a)) return "no ensemble agreement available";
  return `${agreementLabel(a).word}: ${Math.round(a * 100)}% of ${n} parameter settings agree on the direction`;
}
