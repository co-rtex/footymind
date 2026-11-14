/**
 * Canonical pitch metres <-> canvas CSS pixels.
 *
 * Canonical: origin at the centre spot, x to the right along the length, and
 * y UP across the width. Canvas: origin top-left, y DOWN. The transform keeps
 * a uniform scale (no distortion) and centres the pitch with a margin, so that
 * goals and players standing just outside the lines stay visible.
 */
import type { Pitch } from "../api/types";

export interface PitchTransform {
  /** CSS pixels per metre */
  scale: number;
  /** canvas CSS-pixel position of the centre spot */
  cx: number;
  cy: number;
  pitch: Pitch;
  cssWidth: number;
  cssHeight: number;
}

/** Metres of margin drawn around the touchlines/goal lines. */
export const PITCH_MARGIN_M = 3.5;

export function fitPitch(cssWidth: number, cssHeight: number, pitch: Pitch, marginM = PITCH_MARGIN_M): PitchTransform {
  const w = Math.max(1, cssWidth);
  const h = Math.max(1, cssHeight);
  const scale = Math.min(w / (pitch.length + 2 * marginM), h / (pitch.width + 2 * marginM));
  return { scale, cx: w / 2, cy: h / 2, pitch, cssWidth: w, cssHeight: h };
}

/** Canvas height that fits the whole pitch (plus margin) at a given width. */
export function pitchAspect(pitch: Pitch, marginM = PITCH_MARGIN_M): number {
  return (pitch.width + 2 * marginM) / (pitch.length + 2 * marginM);
}

/**
 * Viewport height kept free for the sticky top bar and the transport under the pitch, so that a
 * pitch scrolled to the top of the screen (the guided tour does that on stacked layouts) leaves the
 * play/pause controls visible.
 */
export const PITCH_RESERVED_HEIGHT_PX = 168;

/**
 * CSS width of the pitch canvas: the full container width, unless the pitch would then be too
 * tall for the viewport (tablet landscape, short laptop screens), in which case it is narrowed
 * (and centred) to fit. It never shrinks below 60% of the container or 240 px.
 */
export function pitchCssWidth(containerWidth: number, viewportHeight: number, aspect: number): number {
  const full = Math.max(240, Math.floor(containerWidth));
  if (!(viewportHeight > 0) || !(aspect > 0)) return full;
  const byHeight = Math.floor((viewportHeight - PITCH_RESERVED_HEIGHT_PX) / aspect);
  const floor = Math.max(240, Math.floor(full * 0.6));
  return Math.min(full, Math.max(floor, byHeight));
}

export function toCanvasX(t: PitchTransform, x: number): number {
  return t.cx + x * t.scale;
}

/** Canonical y is up and canvas y is down, hence the minus sign. */
export function toCanvasY(t: PitchTransform, y: number): number {
  return t.cy - y * t.scale;
}

export function toCanvas(t: PitchTransform, x: number, y: number): [number, number] {
  return [toCanvasX(t, x), toCanvasY(t, y)];
}

export function toPitch(t: PitchTransform, px: number, py: number): [number, number] {
  return [(px - t.cx) / t.scale, (t.cy - py) / t.scale];
}

export function metresToPx(t: PitchTransform, m: number): number {
  return m * t.scale;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function insidePitch(pitch: Pitch, x: number, y: number, marginM = 0): boolean {
  return Math.abs(x) <= pitch.length / 2 + marginM && Math.abs(y) <= pitch.width / 2 + marginM;
}
