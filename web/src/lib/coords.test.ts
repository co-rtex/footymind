import { describe, expect, it } from "vitest";
import { fitPitch, insidePitch, pitchAspect, PITCH_MARGIN_M, toCanvas, toCanvasY, toPitch } from "./coords";

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
