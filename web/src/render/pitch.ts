/**
 * Pitch markings in canonical metres (IFAB Law 1 dimensions), drawn through
 * the canonical -> canvas transform.
 */
import { toCanvasX, toCanvasY, type PitchTransform } from "../lib/coords";
import type { CanvasTheme } from "./theme";

export const MARKINGS = {
  centreCircleR: 9.15,
  penaltyAreaDepth: 16.5,
  penaltyAreaWidth: 40.32,
  goalAreaDepth: 5.5,
  goalAreaWidth: 18.32,
  penaltySpot: 11,
  arcR: 9.15,
  goalWidth: 7.32,
  goalDepth: 2,
  cornerR: 1,
} as const;

/** Angle (radians) where the penalty arc meets the penalty-area line. */
export function penaltyArcHalfAngle(): number {
  return Math.acos((MARKINGS.penaltyAreaDepth - MARKINGS.penaltySpot) / MARKINGS.arcR);
}

function rectM(ctx: CanvasRenderingContext2D, t: PitchTransform, x0: number, y0: number, x1: number, y1: number) {
  const ax = toCanvasX(t, Math.min(x0, x1));
  const ay = toCanvasY(t, Math.max(y0, y1));
  ctx.rect(ax, ay, Math.abs(x1 - x0) * t.scale, Math.abs(y1 - y0) * t.scale);
}

export function drawPitch(ctx: CanvasRenderingContext2D, t: PitchTransform, theme: CanvasTheme): void {
  const { length: L, width: W } = t.pitch;
  const hl = L / 2;
  const hw = W / 2;
  const s = t.scale;

  ctx.fillStyle = theme.surround;
  ctx.fillRect(0, 0, t.cssWidth, t.cssHeight);

  // grass with subtle mowing stripes
  ctx.fillStyle = theme.pitch;
  ctx.fillRect(toCanvasX(t, -hl - 1.5), toCanvasY(t, hw + 1.5), (L + 3) * s, (W + 3) * s);
  const bands = 14;
  ctx.fillStyle = theme.pitchStripe;
  for (let i = 0; i < bands; i += 2) {
    const x = -hl + (i * L) / bands;
    ctx.fillRect(toCanvasX(t, x), toCanvasY(t, hw), (L / bands) * s, W * s);
  }

  ctx.strokeStyle = theme.pitchLine;
  ctx.fillStyle = theme.pitchLine;
  ctx.lineWidth = Math.max(1, 0.12 * s);
  ctx.lineJoin = "miter";

  ctx.beginPath();
  rectM(ctx, t, -hl, -hw, hl, hw);
  ctx.moveTo(toCanvasX(t, 0), toCanvasY(t, hw));
  ctx.lineTo(toCanvasX(t, 0), toCanvasY(t, -hw));
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(t.cx, t.cy, MARKINGS.centreCircleR * s, 0, Math.PI * 2);
  ctx.stroke();

  const spotR = Math.max(1.4, 0.22 * s);
  const dot = (x: number, y: number) => {
    ctx.beginPath();
    ctx.arc(toCanvasX(t, x), toCanvasY(t, y), spotR, 0, Math.PI * 2);
    ctx.fill();
  };
  dot(0, 0);

  const arcA = penaltyArcHalfAngle();
  for (const side of [-1, 1] as const) {
    const gx = side * hl;
    ctx.beginPath();
    rectM(ctx, t, gx, -MARKINGS.penaltyAreaWidth / 2, gx - side * MARKINGS.penaltyAreaDepth, MARKINGS.penaltyAreaWidth / 2);
    rectM(ctx, t, gx, -MARKINGS.goalAreaWidth / 2, gx - side * MARKINGS.goalAreaDepth, MARKINGS.goalAreaWidth / 2);
    ctx.stroke();

    const spotX = gx - side * MARKINGS.penaltySpot;
    dot(spotX, 0);

    // arc outside the penalty area, facing the centre of the pitch
    ctx.beginPath();
    const base = side === 1 ? Math.PI : 0;
    ctx.arc(toCanvasX(t, spotX), toCanvasY(t, 0), MARKINGS.arcR * s, base - arcA, base + arcA);
    ctx.stroke();

    // goal frame behind the line
    ctx.save();
    ctx.lineWidth = Math.max(1.2, 0.16 * s);
    ctx.strokeStyle = theme.text;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    rectM(ctx, t, gx, -MARKINGS.goalWidth / 2, gx + side * MARKINGS.goalDepth, MARKINGS.goalWidth / 2);
    ctx.stroke();
    ctx.restore();

    // corner arcs
    for (const ys of [-1, 1] as const) {
      ctx.beginPath();
      const cxp = toCanvasX(t, gx);
      const cyp = toCanvasY(t, ys * hw);
      // canvas angles: 0 = +x, PI/2 = down (canonical -y)
      const start = side === 1 ? (ys === 1 ? Math.PI / 2 : Math.PI) : ys === 1 ? 0 : (3 * Math.PI) / 2;
      ctx.arc(cxp, cyp, MARKINGS.cornerR * s, start, start + Math.PI / 2);
      ctx.stroke();
    }
  }
}
