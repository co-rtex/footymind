import { describe, expect, it } from "vitest";
import {
  fitPitch,
  insidePitch,
  pitchAspect,
  pitchCssWidth,
  PITCH_MARGIN_M,
  PITCH_RESERVED_HEIGHT_PX,
  toCanvas,
  toCanvasY,
  toPitch,
} from "./coords";

const PITCH = { length: 105, width: 68 };

describe("canonical <-> canvas transform", () => {
  const t = fitPitch(1120, 760, PITCH);

  it("maps the centre spot to the canvas centre", () => {
    expect(toCanvas(t, 0, 0)).toEqual([560, 380]);
  });

  it("flips y: canonical y is up, canvas y is down", () => {
    const [, top] = toCanvas(t, 0, 34);
    const [, bottom] = toCanvas(t, 0, -34);
    expect(top).toBeLessThan(380);
    expect(bottom).toBeGreaterThan(380);
    expect(toCanvasY(t, 10)).toBeCloseTo(380 - 10 * t.scale);
  });

  it("keeps x direction: +x is to the right", () => {
    expect(toCanvas(t, 52.5, 0)[0]).toBeGreaterThan(toCanvas(t, -52.5, 0)[0]);
  });

  it("round-trips canonical -> canvas -> canonical", () => {
    for (const [x, y] of [
      [0, 0],
      [52.5, 34],
      [-52.5, -34],
      [11.3, -27.9],
    ] as const) {
      const [px, py] = toCanvas(t, x, y);
      const [bx, by] = toPitch(t, px, py);
      expect(bx).toBeCloseTo(x, 9);
      expect(by).toBeCloseTo(y, 9);
    }
  });

  it("uses a uniform scale that fits the pitch plus margin", () => {
    expect(t.scale).toBeCloseTo(Math.min(1120 / (105 + 2 * PITCH_MARGIN_M), 760 / (68 + 2 * PITCH_MARGIN_M)));
    const [l, top] = toCanvas(t, -52.5, 34);
    const [r, bot] = toCanvas(t, 52.5, -34);
    expect(l).toBeGreaterThan(0);
    expect(top).toBeGreaterThan(0);
    expect(r).toBeLessThan(1120);
    expect(bot).toBeLessThan(760);
  });

  it("letterboxes a tall canvas without distortion", () => {
    const tall = fitPitch(400, 1000, PITCH);
    const [l] = toCanvas(tall, -52.5, 0);
    const [r] = toCanvas(tall, 52.5, 0);
    expect(r - l).toBeCloseTo(105 * tall.scale);
    expect(tall.scale).toBeCloseTo(400 / (105 + 2 * PITCH_MARGIN_M));
  });

  it("computes the aspect ratio for a width-driven canvas", () => {
    expect(pitchAspect(PITCH)).toBeCloseTo((68 + 7) / (105 + 7));
  });

  it("checks inside-pitch with an optional margin", () => {
    expect(insidePitch(PITCH, 52.5, 34)).toBe(true);
    expect(insidePitch(PITCH, 53.5, 0)).toBe(false);
    expect(insidePitch(PITCH, 53.5, 0, 2)).toBe(true);
  });
});

describe("pitch canvas width (pitchCssWidth)", () => {
  const aspect = pitchAspect(PITCH);

  it("uses the full container width when the pitch fits the viewport height", () => {
    expect(pitchCssWidth(910, 900, aspect)).toBe(910); // desktop 1440x900, wide layout
    expect(pitchCssWidth(770, 1180, aspect)).toBe(770); // tablet portrait
    expect(pitchCssWidth(356, 844, aspect)).toBe(356); // phone portrait
  });

  it("narrows a pitch that would be taller than the viewport, keeping room for the transport", () => {
    // tablet landscape 1180x820: the full-width pitch is 757 px tall and pushed the controls off screen
    const w = pitchCssWidth(1130, 820, aspect);
    expect(w).toBeLessThan(1130);
    expect(w * aspect).toBeLessThanOrEqual(820 - PITCH_RESERVED_HEIGHT_PX);
  });

  it("never shrinks below 60% of the container or 240 px, and ignores an unknown viewport", () => {
    expect(pitchCssWidth(812, 390, aspect)).toBe(Math.floor(812 * 0.6));
    expect(pitchCssWidth(300, 200, aspect)).toBe(240);
    expect(pitchCssWidth(0, 800, aspect)).toBe(240);
    expect(pitchCssWidth(1000, 0, aspect)).toBe(1000);
  });
});
