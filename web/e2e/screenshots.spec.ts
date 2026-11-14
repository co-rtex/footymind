import { expect, test, type Page } from "@playwright/test";
import { dragPlayer, fetchAnalysis, openGoalPossession, waitForAnalysis, type Analysis } from "./helpers";

/**
 * README screenshots of the real app on real data.
 *   npm run screenshots                       -> web/screenshots at 2x
 *   SHOT_DPR=1 SHOT_DIR=/tmp/x npm run screenshots   -> quick 1x review
 */
const DPR = Number(process.env.SHOT_DPR ?? 2);
const DIR = process.env.SHOT_DIR ?? "screenshots";

async function pauseOnLaterFrame(page: Page): Promise<number> {
  await waitForAnalysis(page);
  await page.getByTestId("play-toggle").click();
  await page.waitForTimeout(1200);
  await page.getByTestId("play-toggle").click();
  return waitForAnalysis(page);
}

/** An attacker other than the carrier with a defender within 10 m, so "marker follows" has someone to move. */
function pickAttacker(a: Analysis & { attack_sign?: number }) {
  const bx = a.ball_x ?? 0;
  const by = a.ball_y ?? 0;
  // keep clear of the carrier so the ghost and arrow are legible
  const att = a.players.filter(
    (p) =>
      p.team === a.attacking_team &&
      p.x !== null &&
      !p.is_gk &&
      p.player_id !== a.carrier_id &&
      Math.hypot(p.x - bx, p.y! - by) > 10,
  );
  const def = a.players.filter((p) => p.team !== a.attacking_team && p.x !== null);
  const scored = att
    .map((p) => ({ p, d: Math.min(...def.map((q) => Math.hypot(q.x! - p.x!, q.y! - p.y!))) }))
    .filter((s) => s.d < 8 && s.p.reach_radius_m !== null)
    .sort((s, t) => s.d - t.d);
  const pick = scored[0]?.p ?? att[0];
  if (!pick) throw new Error("no attacker");
  return pick;
}

test.describe("desktop", () => {
  test.use({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: DPR });

  test("landing", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("possession-table").locator("tbody tr").first()).toBeVisible();
    // the featured numbers are live; capture once they are in
    await expect(page.getByTestId("featured-card-broadcast").getByTestId("featured-facts")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("featured-shadow-loading")).toHaveCount(0, { timeout: 30_000 });
    await page.screenshot({ path: `${DIR}/landing-desktop.png` });
  });

  test("guided tour, what-if step", async ({ page }) => {
    await page.goto("/#/at/metrica-sample-1?f=53708&tour=press");
    await waitForAnalysis(page);
    await page.getByTestId("guide-step-3").click();
    await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 30_000 });
    await page.mouse.move(0, 0);
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${DIR}/guided-tour-desktop.png` });
  });

  test("guided tour, static vs defender response", async ({ page }) => {
    await page.goto("/#/at/metrica-sample-2?f=31360&tour=response");
    await waitForAnalysis(page);
    await page.getByTestId("guide-step-3").click();
    await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 30_000 });
    await expect(page.getByTestId("policy-comparison")).toBeVisible({ timeout: 30_000 });
    await page.mouse.move(0, 0);
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${DIR}/guided-tour-response-desktop.png` });
  });

  test("lab analysis and scenario", async ({ page }) => {
    await openGoalPossession(page);
    const frame = await pauseOnLaterFrame(page);
    await page.mouse.move(0, 0);
    await page.screenshot({ path: `${DIR}/lab-analysis-desktop.png` });

    const matchId = decodeURIComponent(page.url().split("/lab/")[1] ?? "").split(":")[0] ?? "";
    const a = (await fetchAnalysis(page, matchId, frame)) as Analysis & { attack_sign: number };
    await page.getByRole("radio", { name: "Edit scenario" }).click();
    await page.getByTestId("policy-select").selectOption("marker_follows");
    const p = pickAttacker(a);
    const to = { x: p.x! + a.attack_sign * 6, y: p.y! + (p.y! > 0 ? -3 : 3) };
    await dragPlayer(page, { x: p.x!, y: p.y! }, to);
    await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success");
    await page.mouse.move(0, 0);
    await page.getByRole("radio", { name: "Difference" }).click();
    await page.waitForTimeout(150);
    await page.screenshot({ path: `${DIR}/scenario-difference-desktop.png` });
  });
});

test.describe("desktop, experimental layer", () => {
  test.use({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: DPR });

  test("defensive shadow at the featured frame", async ({ page }) => {
    await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
    await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");
    await page.getByRole("checkbox", { name: "Defensive shadow (experimental)" }).check();
    await expect(page.getByTestId("shadow-panel")).toHaveAttribute("data-frame", "53708", { timeout: 30_000 });
    await page.mouse.move(0, 0);
    await page.screenshot({ path: `${DIR}/defensive-shadow-desktop.png` });
  });
});

test.describe("desktop, broadcast tracking", () => {
  test.use({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: DPR });

  test("SkillCorner possession with observed and extrapolated players", async ({ page }) => {
    const poss = (await (await page.request.get("/api/matches/skillcorner-2017461/possessions?min_passes=3")).json()) as {
      possession_id: string;
    }[];
    await page.goto(`/#/lab/${encodeURIComponent(poss[0]!.possession_id)}`);
    await waitForAnalysis(page);
    await expect(page.getByTestId("legend-estimated")).toBeVisible();
    await page.mouse.move(0, 0);
    await page.screenshot({ path: `${DIR}/broadcast-estimated-desktop.png` });
  });
});

test.describe("narrow", () => {
  test.use({ viewport: { width: 390, height: 844 }, deviceScaleFactor: DPR, isMobile: false, hasTouch: false });

  test("lab on a phone-width screen", async ({ page }) => {
    await openGoalPossession(page);
    await waitForAnalysis(page);
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(width).toBeLessThanOrEqual(390);
    await page.screenshot({ path: `${DIR}/lab-narrow.png`, fullPage: true });
  });
});
