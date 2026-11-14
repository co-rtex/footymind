import { expect, test, type Page } from "@playwright/test";

/**
 * The first-visit journey against the real backend: understand the landing, take the guided
 * tour of the featured situation (watch, understand, what if), and find the supporting panels.
 * Numbers are never asserted as constants: the landing's live numbers must equal the lab's.
 */

const TOUR_URL = /#\/at\/metrica-sample-1\?f=53708&tour=press/;

async function landingFacts(page: Page): Promise<{ obs: string; hyp: string }> {
  const card = page.getByTestId("featured-card-press");
  const facts = card.getByTestId("featured-facts");
  await expect(facts).toBeVisible({ timeout: 30_000 });
  const text = (await facts.textContent()) ?? "";
  const m = text.match(/estimated completion\s*(\d\.\d\d)\s*→\s*to\s*(\d\.\d\d)/);
  expect(m, `pass fact in: ${text}`).not.toBeNull();
  return { obs: m![1]!, hyp: m![2]! };
}

test("landing explains the idea and computes the featured numbers live", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "The Invisible Game", level: 1 })).toBeVisible();
  await expect(page.getByText("Event data says a pass went from")).toBeVisible();
  // the four things event data cannot see, each with its provenance label
  const pillars = page.getByRole("definition").filter({ hasText: /./ });
  expect(await pillars.count()).toBeGreaterThan(3);
  await expect(page.getByTestId("tour-card")).toContainText("A three-step guided tour");
  await expect(page.getByTestId("tour-link")).toHaveAttribute("href", /tour=press/);

  // live numbers, labelled, and the five-label key
  const facts = await landingFacts(page);
  expect(Number(facts.hyp)).not.toBe(Number(facts.obs));
  const card = page.getByTestId("featured-card-press");
  await expect(card.locator(".prov--estimated").first()).toBeVisible();
  await expect(card.locator(".prov--hypothetical")).toBeVisible();
  await expect(card).toContainText(/model settings agree/);
  await expect(page.getByTestId("featured-card-broadcast").getByTestId("featured-facts")).toContainText(
    "estimated (off camera)",
    { timeout: 30_000 },
  );
  const key = page.getByTestId("provenance-key").first();
  for (const label of ["observed", "derived", "estimated", "hypothetical", "simulated"]) {
    await expect(key.locator(`.prov--${label}`)).toBeVisible();
  }
  // the possession list shows a first page and can expand
  const more = page.getByTestId("possessions-more");
  await expect(more).toBeVisible();
  await more.click();
  await expect(more).toBeHidden();
  expect(errors).toEqual([]);
});

test("guided tour: watch, understand, then the what-if, matching the landing's numbers", async ({ page }) => {
  await page.goto("/");
  const facts = await landingFacts(page);
  await page.getByTestId("tour-link").click();
  await expect(page).toHaveURL(TOUR_URL);
  const guide = page.getByTestId("guide");
  await expect(guide).toBeVisible();
  await expect(page.getByTestId("guide-step-1")).toHaveAttribute("aria-current", "step");

  // 1. watch the observed possession
  await page.getByTestId("guide-watch").click();
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Pause");

  // 2. understand: paused on the decision frame with the evidence
  await page.getByTestId("guide-next").click();
  await expect(page.getByTestId("guide-step-2")).toHaveAttribute("aria-current", "step");
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Play");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");
  await expect(page.getByTestId("why-panel")).toHaveAttribute("data-frame", "53708");
  await guide.getByRole("button", { name: "Similar moments" }).click();
  await expect(page.locator("#similar-title")).toBeFocused();
  await expect(page.getByTestId("pitch-key")).toBeVisible();

  // 3. what if: the curated move, compared with what was observed
  await page.getByTestId("guide-next").click();
  await expect(page.getByTestId("guide-step-3")).toHaveAttribute("aria-current", "step");
  await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  const top = page.getByTestId("option-deltas").locator("tbody tr").first();
  await expect(top).toContainText("#18");
  // the landing showed exactly the lab's numbers
  await expect(top.locator("td").nth(0)).toHaveText(facts.obs);
  await expect(top.locator("td").nth(1)).toHaveText(facts.hyp);
  await expect(page.getByRole("radio", { name: "Difference" })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("evidence")).toContainText("not what would have happened in the match");

  // the defender-response selector, and why it moves nobody for a defender-only edit
  await guide.getByRole("button", { name: "Defender response" }).click();
  const select = page.getByTestId("policy-select");
  await expect(select).toBeFocused();
  await select.selectOption("marker_follows");
  await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  await expect(page.getByTestId("response-hint")).toContainText("Move an attacker");

  await page.getByTestId("guide-finish").click();
  await expect(guide).toBeHidden();
});

test("the player card describes position in football terms, with provenance", async ({ page }) => {
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("aria-busy", "false");
  const menu = page.getByTestId("player-menu");
  const id = await menu.locator("option").nth(3).getAttribute("value");
  await menu.selectOption(id!);
  const card = page.getByTestId("player-inspector");
  await expect(card.getByTestId("player-zone")).toHaveText(/(third|penalty area), (left|right) (wing|half-space)|centre/);
  await expect(card.getByTestId("player-zone").locator(".prov")).toBeVisible();
  await expect(card).not.toContainText(id!); // no raw player id
  await expect(card.getByTestId("player-coords")).toBeHidden();
  await card.getByText("Pitch coordinates").click();
  await expect(card.getByTestId("player-coords")).toContainText("from the centre spot");
});

test("analysis jump links, visible column help and the labels key", async ({ page }) => {
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("aria-busy", "false");
  const nav = page.getByTestId("side-nav");
  await nav.getByRole("button", { name: "Similar moments" }).click();
  await expect(page.locator("#similar-title")).toBeFocused();
  await nav.getByRole("button", { name: "Labels" }).click();
  const labels = page.getByTestId("legend-labels");
  await expect(labels).toHaveAttribute("open", "");
  await expect(labels.getByTestId("provenance-key")).toContainText("did not happen");
  const help = page.getByTestId("pass-options-help");
  await help.locator("summary").click();
  await expect(help).toContainText("estimated chance that no defender intercepts");
});

test("keyboard: the guided tour is reachable early and starts with Enter", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("tour-link")).toBeVisible();
  const order: string[] = [];
  for (let i = 0; i < 8; i++) {
    await page.keyboard.press("Tab");
    const t = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      return el?.dataset.testid ?? el?.textContent?.trim().slice(0, 30) ?? "";
    });
    order.push(t);
    if (t === "tour-link") break;
  }
  expect(order[0]).toBe("Skip to content");
  expect(order).toContain("tour-link");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(TOUR_URL);
  await expect(page.getByTestId("guide")).toBeVisible();
});

test("featured cards explain a failed live computation and offer retry", async ({ page }) => {
  await page.route("**/api/scenarios", (route) => route.fulfill({ status: 503, body: "" }));
  await page.goto("/");
  const card = page.getByTestId("featured-card-press");
  await expect(card.getByText("Live numbers unavailable")).toBeVisible({ timeout: 30_000 });
  await expect(card.getByRole("button", { name: "Retry" })).toBeVisible();
  // the story and the links stay usable
  await expect(card.getByTestId("featured-link")).toBeVisible();
  await page.unroute("**/api/scenarios");
  await card.getByRole("button", { name: "Retry" }).click();
  await expect(card.getByTestId("featured-facts")).toBeVisible({ timeout: 30_000 });
});
