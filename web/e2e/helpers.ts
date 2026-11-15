import { expect, type Page } from "@playwright/test";

/** Mirror of src/lib/coords.ts PITCH_MARGIN_M. Kept independent so the test checks the real transform. */
const MARGIN_M = 3.5;
const PITCH = { length: 105, width: 68 };

export interface AnalysisPlayer {
  player_id: string;
  team: "home" | "away";
  jersey: number;
  x: number | null;
  y: number | null;
  is_gk: boolean;
  reach_origin_x: number | null;
  reach_origin_y: number | null;
  reach_radius_m: number | null;
}

export interface Analysis {
  frame: number;
  attacking_team: "home" | "away" | null;
  ball_x: number | null;
  ball_y: number | null;
  carrier_id: string | null;
  players: AnalysisPlayer[];
  pass_options: unknown[];
}

/** Canonical pitch metres (y up) -> page CSS pixels (y down). */
export async function pitchToScreen(page: Page, x: number, y: number): Promise<{ x: number; y: number }> {
  const box = await page.getByTestId("pitch-canvas").boundingBox();
  if (!box) throw new Error("pitch canvas not visible");
  const scale = Math.min(box.width / (PITCH.length + 2 * MARGIN_M), box.height / (PITCH.width + 2 * MARGIN_M));
  return { x: box.x + box.width / 2 + x * scale, y: box.y + box.height / 2 - y * scale };
}

export async function openGoalPossession(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "The Invisible Game", level: 1 })).toBeVisible();
  const table = page.getByTestId("possession-table");
  await expect(table).toBeVisible();
  // the list shows its first page; goals end in a shot, so filter to reach one
  await page.getByLabel("Ends in a shot").check();
  const goalRow = table.locator("tbody tr", { has: page.locator(".badge--goal") }).first();
  await goalRow.getByRole("link").click();
  await expect(page.getByTestId("pitch-canvas")).toBeVisible();
}

export async function waitForAnalysis(page: Page): Promise<number> {
  const panel = page.getByTestId("analysis-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toHaveAttribute("aria-busy", "false");
  const frame = Number(await panel.getAttribute("data-frame"));
  expect(Number.isFinite(frame)).toBe(true);
  return frame;
}

export async function fetchAnalysis(page: Page, matchId: string, frame: number): Promise<Analysis> {
  const res = await page.request.get(`/api/matches/${matchId}/frames/${frame}/analysis?grid=false`);
  expect(res.ok()).toBe(true);
  return (await res.json()) as Analysis;
}

/** A defending outfield player near the ball, preferring one whose reach region is known. */
export function pickDefender(a: Analysis): AnalysisPlayer {
  const defending = a.players.filter(
    (p) => p.team !== a.attacking_team && p.x !== null && p.y !== null && !p.is_gk && p.reach_radius_m !== null,
  );
  const bx = a.ball_x ?? 0;
  const by = a.ball_y ?? 0;
  defending.sort((p, q) => Math.hypot(p.x! - bx, p.y! - by) - Math.hypot(q.x! - bx, q.y! - by));
  const d = defending[0];
  if (!d) throw new Error("no defender with a known reach region");
  return d;
}

export async function dragPlayer(page: Page, from: { x: number; y: number }, to: { x: number; y: number }) {
  const a = await pitchToScreen(page, from.x, from.y);
  const b = await pitchToScreen(page, to.x, to.y);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 6 });
  await expect(page.getByTestId("drag-warning")).toBeVisible();
  await page.mouse.move(b.x, b.y, { steps: 6 });
  await page.mouse.up();
}
