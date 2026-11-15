/**
 * Sub-frame interpolation of tracking positions with strict null handling.
 *
 * Rule: a position is drawn only if the NEAREST frame observed it. Between two
 * observed samples we interpolate linearly, which is display smoothing and
 * never invents a position where the provider had none. If only one of the two
 * neighbouring frames is observed, we use it only when it is the nearer one.
 * We never interpolate across a period boundary.
 */

export interface Vec2 {
  x: number;
  y: number;
}

/** Result of sampling a track: nothing drawable, an observed position, or a provider estimate. */
export const NOT_DRAWN = 0;
export const OBSERVED = 1;
export const ESTIMATED = 2;
export type SampleKind = typeof NOT_DRAWN | typeof OBSERVED | typeof ESTIMATED;

/**
 * Sample item `k` of a flat [frame][item][x|y] array at fractional frame index `t`,
 * and report its provenance.
 *
 * `detected` ([frame][item], 1 = detected, 0 = provider extrapolation) is
 * optional; without it every drawn position is observed. A sample is
 * ESTIMATED if ANY frame that contributes to it is an extrapolation, so an
 * estimated position is never blended into an observed one unmarked. A frame
 * with zero interpolation weight does not contribute.
 */
export function sampleTrack(
  arr: Float32Array,
  nItems: number,
  nFrames: number,
  t: number,
  k: number,
  out: Vec2,
  period?: ArrayLike<number>,
  detected?: Uint8Array | null,
): SampleKind {
  if (nFrames <= 0 || !Number.isFinite(t)) return NOT_DRAWN;
  const tc = t < 0 ? 0 : t > nFrames - 1 ? nFrames - 1 : t;
  const i0 = Math.floor(tc);
  const i1 = i0 + 1 < nFrames ? i0 + 1 : i0;
  const a = tc - i0;
  const stride = nItems * 2;
  const b0 = i0 * stride + k * 2;
  const b1 = i1 * stride + k * 2;
  const x0 = arr[b0] as number;
  const y0 = arr[b0 + 1] as number;
  const x1 = arr[b1] as number;
  const y1 = arr[b1 + 1] as number;
  const ok0 = Number.isFinite(x0) && Number.isFinite(y0);
  const ok1 = Number.isFinite(x1) && Number.isFinite(y1);
  const samePeriod = !period || period[i0] === period[i1];
  const est = (i: number) => (detected ? detected[i * nItems + k] === 0 : false);
  if (ok0 && ok1 && samePeriod) {
    out.x = x0 + (x1 - x0) * a;
    out.y = y0 + (y1 - y0) * a;
    const uses1 = a > 0 && i1 !== i0;
    return est(i0) || (uses1 && est(i1)) ? ESTIMATED : OBSERVED;
  }
  const nearestIs0 = a < 0.5 || i1 === i0;
  if (nearestIs0 && ok0) {
    out.x = x0;
    out.y = y0;
    return est(i0) ? ESTIMATED : OBSERVED;
  }
  if (!nearestIs0 && ok1) {
    out.x = x1;
    out.y = y1;
    return est(i1) ? ESTIMATED : OBSERVED;
  }
  return NOT_DRAWN;
}

/**
 * Sample item `k` of a flat [frame][item][x|y] array at fractional frame index `t`.
 * Writes into `out` and returns true, or returns false for "not observed".
 */
export function sampleXY(
  arr: Float32Array,
  nItems: number,
  nFrames: number,
  t: number,
  k: number,
  out: Vec2,
  period?: ArrayLike<number>,
): boolean {
  return sampleTrack(arr, nItems, nFrames, t, k, out, period) !== NOT_DRAWN;
}

/** True when either end of the segment between frames i-1 and i is a provider extrapolation. */
export function segmentEstimated(detected: Uint8Array | null | undefined, nItems: number, i: number, k: number): boolean {
  if (!detected || i <= 0) return false;
  return detected[(i - 1) * nItems + k] === 0 || detected[i * nItems + k] === 0;
}

/** Observed value at an integer frame index, or null. */
export function valueAt(arr: Float32Array, nItems: number, i: number, k: number): Vec2 | null {
  const b = i * nItems * 2 + k * 2;
  const x = arr[b];
  const y = arr[b + 1];
  if (x === undefined || y === undefined || !Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

/**
 * Speed from a centred finite difference over +-h frames (derived quantity).
 * Returns null when either end is not observed.
 */
export function finiteDiffSpeed(
  arr: Float32Array,
  nItems: number,
  nFrames: number,
  i: number,
  k: number,
  dtPerFrame: number,
  h = 2,
): number | null {
  const lo = Math.max(0, i - h);
  const hi = Math.min(nFrames - 1, i + h);
  if (hi === lo) return null;
  const a = valueAt(arr, nItems, lo, k);
  const b = valueAt(arr, nItems, hi, k);
  if (!a || !b) return null;
  return Math.hypot(b.x - a.x, b.y - a.y) / ((hi - lo) * dtPerFrame);
}
