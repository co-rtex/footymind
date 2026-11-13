/**
 * SYNTHETIC test fixtures. These are NOT real tracking data: values are made
 * up to exercise parsing, null handling and transforms. Real-data behaviour
 * is covered by the Playwright e2e suite against the running backend.
 */
import type { FrameAnalysisOut, Replay } from "../api/types";

export const SYNTHETIC_SOURCE = {
  provider: "synthetic",
  name: "Synthetic fixture (not real data)",
  url: "https://example.invalid/synthetic",
  license_note: "Synthetic test data generated for unit tests.",
};

/** 3 frames x 2 players. Player 1 is unobserved in frame 1; the ball is unobserved in frame 2. */
export function syntheticReplay(overrides: Partial<Replay> = {}): Replay {
  return {
    match_id: "synthetic-1",
    fps: 25,
    step: 1,
    pitch: { length: 105, width: 68 },
    players: [
      { player_id: "home_1", team: "home", jersey: 1 },
      { player_id: "away_2", team: "away", jersey: 2 },
    ],
    frame: [100, 101, 102],
    time_s: [4, 4.04, 4.08],
    period: [1, 1, 1],
    // [frame][player][x,y]
    players_xy: [0, 0, 10, -5, 1, 1, null, null, 2, 2, 12, -5],
    ball_xy: [0.5, 0.5, 1.5, 1.5, null, null],
    events: [],
    possession: null,
    attack_sign: { "1": { home: 1, away: -1 }, "2": { home: -1, away: 1 } },
    provenance: "observed",
    source: SYNTHETIC_SOURCE,
    ...overrides,
  };
}

export function syntheticAnalysis(overrides: Partial<FrameAnalysisOut> = {}): FrameAnalysisOut {
  return {
    match_id: "synthetic-1",
    frame: 101,
    period: 1,
    time_s: 4.04,
    attacking_team: "home",
    attack_sign: 1,
    possession_id: "synthetic-1:p0000",
    carrier_id: "home_1",
    ball_x: 1.5,
    ball_y: 1.5,
    ball_provenance: "observed",
    players: [],
    pitch_control: { nx: 2, ny: 2, x0: -26.25, y0: -17, dx: 52.5, dy: 34, values: [0.1, 0.2, 0.8, 0.9], provenance: "estimated" },
    pass_options: [],
    pressure: null,
    control_area_m2: 100,
    control_area_final_third_m2: 10,
    model: { name: "synthetic", reference: "none", status: "synthetic", params: {} },
    option_model: null,
    warnings: [],
    ...overrides,
  };
}
