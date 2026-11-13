import { expect, test } from "@playwright/test";
import { dragPlayer, fetchAnalysis, openGoalPossession, pickDefender, waitForAnalysis } from "./helpers";

/**
 * Flagship flow against the real backend and real Metrica tracking data:
 * landing -> possession -> play -> pause -> analysis overlay -> drag a
 * defender -> scenario deltas -> difference view -> reset.
 */
test("replay, analyse and edit a scenario on real tracking data", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await openGoalPossession(page);
  await expect(page.getByTestId("source-attribution")).toContainText("Metrica Sports");

  // The lab opens paused on the first frame of the possession with analysis.
  const firstFrame = await waitForAnalysis(page);
  const clock = page.getByTestId("clock");
  const clockBefore = await clock.textContent();

  // Play: the canvas must change and the clock must advance.
  const canvas = page.getByTestId("pitch-canvas");
  const shotA = await canvas.screenshot();
  await page.getByTestId("play-toggle").click();
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Pause");
  await page.waitForTimeout(1600);
  const shotB = await canvas.screenshot();
  expect(Buffer.compare(shotA, shotB)).not.toBe(0);
  await expect(clock).not.toHaveText(clockBefore ?? "");

  // Pause with the keyboard shortcut, then step one frame with the arrow key.
  await page.locator("body").press("Space");
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Play");
  await page.locator("body").press("ArrowRight");

  // Analysis overlay for the paused frame.
  const frame = await waitForAnalysis(page);
  expect(frame).toBeGreaterThan(firstFrame);
  await expect(page.getByTestId("pitch-status")).toHaveText(/model overlay/);
  await expect(page.getByText("Estimated pitch control (model, uncalibrated)")).toBeVisible();
  await expect(page.getByTestId("model-key")).toContainText(/Option model \(calibrated on [\d,]+ real passes\)/);
  await expect(page.getByTestId("model-key")).toContainText("Physics heuristic (uncalibrated)");
  const optionRows = page.getByTestId("pass-options").locator("tbody tr");
  expect(await optionRows.count()).toBeGreaterThan(0);

  // Edit scenario: drag the defender nearest the ball 3 m towards the ball.
  const matchId = decodeURIComponent(page.url().split("/lab/")[1] ?? "").split(":")[0] ?? "";
  const a = await fetchAnalysis(page, matchId, frame);
  expect(a.attacking_team).not.toBeNull();
  const d = pickDefender(a);
  await page.getByRole("radio", { name: "Edit scenario" }).click();
  await expect(page.getByTestId("scenario-panel")).toBeVisible();
  const bx = a.ball_x ?? 0;
  const by = a.ball_y ?? 0;
  const len = Math.hypot(bx - d.x!, by - d.y!) || 1;
  const step = Math.min(3, len * 0.6);
  const to = { x: d.x! + ((bx - d.x!) / len) * step, y: d.y! + ((by - d.y!) / len) * step };
  await dragPlayer(page, { x: d.x!, y: d.y! }, to);

  // Scenario response: deltas, interval bars, agreement, evidence.
  const panel = page.getByTestId("scenario-panel");
  await expect(panel).toHaveAttribute("data-status", "success");
  const deltas = page.getByTestId("option-deltas");
  await expect(deltas).toBeVisible();
  const rows = deltas.locator("tbody tr");
  expect(await rows.count()).toBeGreaterThan(0);
  await expect(rows.first().locator("td").nth(2)).toHaveText(/^[+−]?\d\.\d\d$|^0\.00$/);
  await expect(deltas.locator("svg.ibars").first()).toBeVisible();
  await expect(page.getByTestId("scenario-headline")).toContainText("parameter settings agree");
  await expect(page.getByTestId("scenario-verdict")).toBeVisible();
  const evidence = page.getByTestId("evidence");
  await expect(evidence).toContainText("Evidence & assumptions");
  await expect(evidence).toContainText("not what would have happened in the match");
  await expect(panel.getByText(`#${d.jersey} ${d.team === "home" ? "Home" : "Away"} to (`)).toBeVisible();

  // Difference view, then response policy toggle re-runs the scenario.
  await page.getByRole("radio", { name: "Difference" }).click();
  await expect(page.getByText(/hypothetical − observed\), model, uncalibrated/)).toBeVisible();
  await expect(page.getByTestId("delta-model")).toContainText("Option model (calibrated on");
  await page.getByRole("radio", { name: "Physics heuristic" }).click();
  await expect(panel).toHaveAttribute("data-status", "success");
  await expect(page.getByTestId("delta-model")).toContainText("Physics heuristic (uncalibrated)");
  await page.getByRole("radio", { name: "Marker follows" }).click();
  await expect(panel).toHaveAttribute("data-status", "success");

  // Reset returns to the observed state.
  await page.getByTestId("reset-scenario").click();
  await expect(page.getByTestId("option-deltas")).toHaveCount(0);
  await expect(panel).toContainText("Drag any player");

  expect(errors).toEqual([]);
});

test("possession list loads with attribution and filters", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("source-attribution")).toContainText("Metrica Sports");
  const rows = page.getByTestId("possession-table").locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  const all = await rows.count();
  await page.getByLabel("Ends in a shot").check();
  await expect.poll(() => rows.count()).toBeLessThan(all);
  const shots = await rows.count();
  expect(shots).toBeGreaterThan(0);
  await expect(rows.locator(".badge--shot, .badge--goal")).toHaveCount(shots);
});
