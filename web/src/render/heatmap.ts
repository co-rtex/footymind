/**
 * Pitch-control rasters. A grid is painted once into an nx-by-ny offscreen
 * canvas (one pixel per cell) and then scaled onto the pitch with bilinear
 * smoothing, so drawing it each frame costs one drawImage call.
 */
import type { PitchControlOut } from "../api/types";
import { hexToRgb, type RGB } from "./theme";

export interface GridGeom {
  nx: number;
  ny: number;
  x0: number;
  y0: number;
  dx: number;
  dy: number;
}

/** Map control in [0, 1] to a diverging colour: defending colour -> transparent at 0.5 -> attacking colour. */
export function controlColour(v: number, att: RGB, def: RGB, maxAlpha: number): [number, number, number, number] {
  if (!Number.isFinite(v)) return [0, 0, 0, 0];
  const d = (v - 0.5) * 2; // -1..1
  const c = d >= 0 ? att : def;
  // slightly convex ramp: contested space stays nearly clear, confident control reads clearly
  const a = Math.min(1, Math.abs(d)) ** 1.15 * maxAlpha;
  return [c.r, c.g, c.b, Math.round(a * 255)];
}

/** Map a control delta in [-1, 1] to gain / loss colours with a perceptual ramp. */
export function deltaColour(d: number, gain: RGB, loss: RGB, maxAlpha: number): [number, number, number, number] {
  if (!Number.isFinite(d) || Math.abs(d) < 0.01) return [0, 0, 0, 0];
  const c = d > 0 ? gain : loss;
  const a = Math.min(1, Math.abs(d) / 0.45) ** 0.7 * maxAlpha;
  return [c.r, c.g, c.b, Math.round(a * 255)];
}

/** Row index in the raster for grid row j: grid rows run from y = -W/2 upwards, canvas rows run downwards. */
export function rasterRow(j: number, ny: number): number {
  return ny - 1 - j;
}

export function paintGrid(
  geom: GridGeom,
  values: ArrayLike<number>,
  colour: (v: number) => [number, number, number, number],
): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const cv = document.createElement("canvas");
  cv.width = geom.nx;
  cv.height = geom.ny;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(geom.nx, geom.ny);
  for (let j = 0; j < geom.ny; j++) {
    const row = rasterRow(j, geom.ny);
    for (let i = 0; i < geom.nx; i++) {
      const v = values[j * geom.nx + i] as number;
      const [r, g, b, a] = colour(v);
      const o = (row * geom.nx + i) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

export function controlRaster(pc: PitchControlOut, attHex: string, defHex: string, maxAlpha = 0.5) {
  const att = hexToRgb(attHex);
  const def = hexToRgb(defHex);
  return paintGrid(pc, pc.values, (v) => controlColour(v, att, def, maxAlpha));
}

export function sameGeom(a: GridGeom, b: GridGeom): boolean {
  return a.nx === b.nx && a.ny === b.ny && a.x0 === b.x0 && a.y0 === b.y0 && a.dx === b.dx && a.dy === b.dy;
}

/** hypothetical - observed, per cell. Returns null if the grids are not aligned. */
export function controlDelta(obs: PitchControlOut, hyp: PitchControlOut): Float32Array | null {
  if (!sameGeom(obs, hyp) || obs.values.length !== hyp.values.length) return null;
  const out = new Float32Array(obs.values.length);
  for (let i = 0; i < out.length; i++) out[i] = (hyp.values[i] as number) - (obs.values[i] as number);
  return out;
}

export function deltaRaster(geom: GridGeom, delta: Float32Array, gainHex: string, lossHex: string, maxAlpha = 0.75) {
  const gain = hexToRgb(gainHex);
  const loss = hexToRgb(lossHex);
  return paintGrid(geom, delta, (d) => deltaColour(d, gain, loss, maxAlpha));
}
