/**
 * Football description of a position, for the player card (instead of raw coordinates).
 *
 * The zone grid is a product convention, not a model output:
 *   - along the pitch: thirds of the pitch length, from the player's own team's point of view
 *     (defensive, middle, attacking), with the penalty areas named on their own;
 *   - across the pitch: five channels. The centre is the width of the six-yard box
 *     (18.32 m), the half-spaces run to the edge of the penalty area (40.32 m wide), and the
 *     wings are outside it. Left and right are as seen by the player's team when attacking.
 * Penalty-area dimensions are the Laws of the Game values; the pitch length and width come
 * from the match data. The description is derived from the position, so it inherits that
 * position's provenance (observed, estimated or hypothetical).
 */

/** Half-widths of the six-yard box and penalty area, and the penalty-area depth (metres). */
const SIX_HALF = 9.16;
const BOX_HALF = 20.16;
const BOX_DEPTH = 16.5;

export interface PitchZone {
  /** "Attacking third", "Opponent's penalty area", ... */
  area: string;
  /** "left half-space", "centre", ... */
  channel: string;
  /** distance to the centre of the goal the player's team attacks, metres */
  toGoal: number;
}

/**
 * @param x,y canonical metres from the centre spot (y positive towards the top touchline)
 * @param sign +1 if the player's team attacks towards +x in this period, −1 otherwise; 0 or
 *   unknown returns null (left/right and thirds cannot be named without a direction)
 */
export function pitchZone(
  x: number | null,
  y: number | null,
  sign: number | null | undefined,
  pitch: { length: number; width: number },
): PitchZone | null {
  if (x === null || y === null || !Number.isFinite(x) || !Number.isFinite(y) || (sign !== 1 && sign !== -1)) return null;
  const half = pitch.length / 2;
  const ax = sign * x; // towards the goal the team attacks
  const left = sign * y; // facing +ax, positive y is to the left
  const third = pitch.length / 6;
  let area: string;
  if (ax >= half - BOX_DEPTH && Math.abs(y) <= BOX_HALF) area = "Opponent's penalty area";
  else if (ax <= -(half - BOX_DEPTH) && Math.abs(y) <= BOX_HALF) area = "Own penalty area";
  else if (ax > third) area = "Attacking third";
  else if (ax < -third) area = "Defensive third";
  else area = "Middle third";
  const side = left >= 0 ? "left" : "right";
  const a = Math.abs(y);
  const channel = a <= SIX_HALF ? "centre" : a <= BOX_HALF ? `${side} half-space` : `${side} wing`;
  return { area, channel, toGoal: Math.hypot(half - ax, y) };
}

export function describeZone(z: PitchZone): string {
  return `${z.area}, ${z.channel}`;
}
