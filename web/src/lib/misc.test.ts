import { describe, expect, it } from "vitest";
import { controlColour, controlDelta, deltaColour, rasterRow } from "../render/heatmap";
import { syntheticAnalysis } from "../test/fixtures";
import { agreementLabel, agreementSentence } from "./agreement";
import { fmtClock, fmtSigned, matchClockSeconds } from "./format";
import { matchOfPossession, parseRoute, routeHash } from "./hooks";
import { Playback } from "./playback";
import { indexOfFrame } from "./replayIndex";

describe("format", () => {
  it("never prints a signed zero", () => {
    expect(fmtSigned(-0.001)).toBe("0.00");
    expect(fmtSigned(0.004)).toBe("0.00");
    expect(fmtSigned(0.1)).toBe("+0.10");
    expect(fmtSigned(-0.1023)).toBe("−0.10");
    expect(fmtSigned(null)).toBe("n/a");
  });

  it("builds a football match clock per period", () => {
    const periods = [
      { period: 1, first_frame: 1, last_frame: 71268, duration_s: 2850.72, home_attacks_positive_x: true },
      { period: 2, first_frame: 71269, last_frame: 145006, duration_s: 2949.52, home_attacks_positive_x: false },
    ];
    expect(fmtClock(matchClockSeconds(0.04, 1, periods, 25))).toBe("00:00");
    // P2 starts at 45:00
    expect(fmtClock(matchClockSeconds(71269 / 25, 2, periods, 25))).toBe("45:00");
    expect(fmtClock(matchClockSeconds(71269 / 25 + 75, 2, periods, 25))).toBe("46:15");
  });
});

describe("agreement wording", () => {
  it("labels robustness and ignores negligible deltas", () => {
    expect(agreementLabel(0.96).word).toBe("robust");
    expect(agreementLabel(0.75).word).toBe("mixed");
    expect(agreementLabel(0.4).word).toBe("sensitive");
    expect(agreementLabel(0.46, 0.0).word).toBe("no change");
    expect(agreementSentence(0.96, 24, -0.1)).toBe("robust: 96% of 24 parameter settings agree on the direction");
  });
});

describe("heatmap helpers", () => {
  it("flips grid rows (y up) into raster rows (y down)", () => {
    expect(rasterRow(0, 32)).toBe(31);
    expect(rasterRow(31, 32)).toBe(0);
  });

  it("is transparent where control is contested and tinted by the dominant side", () => {
    const att = { r: 255, g: 0, b: 0 };
    const def = { r: 0, g: 0, b: 255 };
    expect(controlColour(0.5, att, def, 0.5)[3]).toBe(0);
    expect(controlColour(1, att, def, 0.5)).toEqual([255, 0, 0, 128]);
    expect(controlColour(0, att, def, 0.5)).toEqual([0, 0, 255, 128]);
    expect(controlColour(NaN, att, def, 0.5)[3]).toBe(0);
    expect(deltaColour(0.001, att, def, 1)[3]).toBe(0);
  });

  it("computes hypothetical minus observed only for aligned grids", () => {
    const o = syntheticAnalysis().pitch_control!;
    const h = { ...o, values: [0.2, 0.2, 0.7, 1] };
    expect(Array.from(controlDelta(o, h)!).map((v) => Number(v.toFixed(3)))).toEqual([0.1, 0, -0.1, 0.1]);
    expect(controlDelta(o, { ...h, nx: 4, ny: 1 })).toBeNull();
  });
});

describe("routing", () => {
  it("round-trips possession ids containing colons", () => {
    const h = routeHash({ name: "lab", possessionId: "metrica-sample-1:p0247" });
    expect(parseRoute(h)).toEqual({ name: "lab", possessionId: "metrica-sample-1:p0247" });
    expect(matchOfPossession("metrica-sample-1:p0247")).toBe("metrica-sample-1");
    expect(parseRoute("#/match/metrica-sample-2")).toEqual({ name: "home", matchId: "metrica-sample-2" });
    expect(parseRoute("")).toEqual({ name: "home", matchId: null });
  });

  it("finds replay indices by frame", () => {
    const frames = Int32Array.from([10, 12, 14, 16]);
    expect(indexOfFrame(frames, 5)).toBe(0);
    expect(indexOfFrame(frames, 13)).toBe(2);
    expect(indexOfFrame(frames, 14)).toBe(2);
    expect(indexOfFrame(frames, 99)).toBe(3);
  });
});

describe("Playback clock", () => {
  function harness(n = 101, fps = 25) {
    let now = 0;
    const queue: FrameRequestCallback[] = [];
    const pb = new Playback(n, fps, 1, {
      now: () => now,
      raf: (cb) => (queue.push(cb), queue.length),
      caf: () => void queue.splice(0),
    });
    const advance = (ms: number) => {
      now += ms;
      const cb = queue.shift();
      cb?.(now);
    };
    return { pb, advance };
  }

  it("advances by elapsed time x fps x speed, independent of refresh rate", () => {
    const { pb, advance } = harness();
    pb.play();
    advance(40);
    expect(pb.t).toBeCloseTo(1);
    pb.setSpeed(2);
    advance(40);
    expect(pb.t).toBeCloseTo(3);
  });

  it("snaps to a whole frame on pause and notifies listeners", () => {
    const { pb, advance } = harness();
    const states: boolean[] = [];
    pb.onState((p) => states.push(p));
    pb.play();
    advance(30);
    pb.pause();
    expect(pb.t).toBe(1);
    expect(states).toEqual([true, false]);
  });

  it("stops at the last frame and restarts from the beginning", () => {
    const { pb, advance } = harness(5);
    pb.play();
    for (let i = 0; i < 10; i++) advance(100);
    expect(pb.playing).toBe(false);
    expect(pb.t).toBe(4);
    pb.play();
    expect(pb.t).toBe(0);
  });

  it("clamps seeks and steps", () => {
    const { pb } = harness(10);
    pb.seek(-3);
    expect(pb.t).toBe(0);
    pb.step(4);
    expect(pb.t).toBe(4);
    pb.step(100);
    expect(pb.t).toBe(9);
  });
});
