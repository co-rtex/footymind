import { describe, expect, it } from "vitest";
import { finiteDiffSpeed, sampleXY, valueAt, type Vec2 } from "./interp";

// SYNTHETIC: 4 frames x 2 players, NaN = not observed.
const N = NaN;
const xy = new Float32Array([
  /* f0 */ 0, 0, 10, 10,
  /* f1 */ 2, 4, N, N,
  /* f2 */ 4, 8, 14, 10,
  /* f3 */ N, N, 16, 10,
]);
const P = 2;
const F = 4;

function at(t: number, k: number, period?: number[]): Vec2 | null {
  const out = { x: 0, y: 0 };
  return sampleXY(xy, P, F, t, k, out, period) ? out : null;
}

describe("sampleXY", () => {
  it("returns exact values on whole frames", () => {
    expect(at(1, 0)).toEqual({ x: 2, y: 4 });
  });

  it("interpolates linearly between two observed frames", () => {
    expect(at(0.5, 0)).toEqual({ x: 1, y: 2 });
    expect(at(1.25, 0)).toEqual({ x: 2.5, y: 5 });
  });

  it("never fabricates a position when the nearest frame is unobserved", () => {
    expect(at(1, 1)).toBeNull();
    expect(at(0.6, 1)).toBeNull(); // nearest is f1 (null)
    expect(at(1.4, 1)).toBeNull(); // nearest is f1 (null)
    expect(at(3, 0)).toBeNull();
  });

  it("holds the nearest observed sample next to a gap instead of interpolating across it", () => {
    expect(at(0.3, 1)).toEqual({ x: 10, y: 10 }); // nearest f0 observed, f1 missing
    expect(at(1.8, 1)).toEqual({ x: 14, y: 10 }); // nearest f2 observed
    expect(at(2.2, 0)).toEqual({ x: 4, y: 8 }); // f3 missing for player 0
    expect(at(2.7, 0)).toBeNull();
  });

  it("does not interpolate across a period boundary", () => {
    const period = [1, 1, 2, 2];
    expect(at(1.4, 0, period)).toEqual({ x: 2, y: 4 });
    expect(at(1.6, 0, period)).toEqual({ x: 4, y: 8 });
  });

  it("clamps t to the window and rejects non-finite t", () => {
    expect(at(-5, 0)).toEqual({ x: 0, y: 0 });
    expect(at(99, 1)).toEqual({ x: 16, y: 10 });
    expect(at(NaN, 0)).toBeNull();
    expect(sampleXY(new Float32Array(0), 1, 0, 0, 0, { x: 0, y: 0 })).toBe(false);
  });
});

describe("valueAt and finiteDiffSpeed", () => {
  it("returns null for unobserved values", () => {
    expect(valueAt(xy, P, 1, 1)).toBeNull();
    expect(valueAt(xy, P, 2, 1)).toEqual({ x: 14, y: 10 });
  });

  it("computes a centred difference speed and refuses gaps", () => {
    // player 0 frames 0 -> 2: (4, 8) over 2 frames of 0.04 s
    expect(finiteDiffSpeed(xy, P, F, 1, 0, 0.04, 1)).toBeCloseTo(Math.hypot(4, 8) / 0.08);
    // player 0, frame 2 +- 1 touches frame 3 (missing)
    expect(finiteDiffSpeed(xy, P, F, 2, 0, 0.04, 1)).toBeNull();
  });
});
