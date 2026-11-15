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

test("landing explains the idea and computes the featured numbers live", { tag: "@skillcorner" }, async ({ page }) => {
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

interface Row {
  obs: string;
  hyp: string;
  delta: string;
}

/** The response card's two rows (static defenders, then the response), read from the live card. */
async function responseFacts(page: Page): Promise<{ static: Row; response: Row }> {
  const cmp = page.getByTestId("featured-card-response").getByTestId("featured-response");
  await expect(cmp).toBeVisible({ timeout: 60_000 });
  const read = async (key: string): Promise<Row> => {
    const text = (await cmp.getByTestId(`featured-response-${key}`).textContent()) ?? "";
    const m = text.match(/(\d\.\d\d)\s*→\s*to\s*(\d\.\d\d)\s*([+−]\d\.\d\d|0\.00)/);
    expect(m, `${key} row in: ${text}`).not.toBeNull();
    return { obs: m![1]!, hyp: m![2]!, delta: m![3]! };
  };
  return { static: await read("static"), response: await read("response") };
}

test("response situation: the conclusion depends on the defender response, and the tour shows why", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto("/");
  const card = page.getByTestId("featured-card-response");
  await expect(card).toContainText("model-based illustration, not what happened in the match");
  const facts = await responseFacts(page);
  // the live numbers differ between the two assumptions, and the simulated move is labelled
  expect(facts.response.hyp).not.toBe(facts.static.hyp);
  expect(facts.response.obs).toBe(facts.static.obs);
  const sim = card.getByTestId("featured-simulated-move");
  await expect(sim).toContainText(/#\d+ Away moved \d+\.\d m by the Marker follows response/);
  await expect(sim.locator(".prov--simulated")).toBeVisible();
  await expect(card.locator(".prov--hypothetical")).toBeVisible();
  await expect(card.getByText("Defensive Shadow")).toHaveCount(0);

  await page.getByTestId("tour-link-response").click();
  await expect(page).toHaveURL(/#\/at\/metrica-sample-2\?f=31360&tour=response/);
  const guide = page.getByTestId("guide");
  await expect(guide).toContainText("One run, two assumptions about the defence");
  await page.getByTestId("guide-next").click();
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "31360");
  await expect(page.getByTestId("guide-next")).toContainText("Next: move #5 Home");
  await page.getByTestId("guide-next").click();
  await expect(page.getByTestId("guide-step-3")).toHaveAttribute("aria-current", "step");

  // step 3: the attacker's move, with the defender response preselected and compared with static defenders
  const panel = page.getByTestId("scenario-panel");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 60_000 });
  await expect(page.getByTestId("policy-select")).toHaveValue("marker_follows");
  await expect(guide.getByTestId("guide-response")).toContainText("Marker follows");
  await expect(page.getByTestId("simulated-move")).toContainText("by the Marker follows response");
  const compare = page.getByTestId("policy-comparison");
  await expect(compare).toBeVisible({ timeout: 60_000 });
  await expect(compare).toHaveAttribute("data-verdict", "changes");
  const row = compare.locator("tbody tr", { hasText: "#5 Home" });
  // the landing showed exactly the lab's numbers, under both assumptions
  await expect(row.locator("td").nth(0)).toHaveText(facts.static.delta);
  await expect(row.locator("td").nth(1)).toHaveText(facts.response.delta);
  const responseRow = page.getByTestId("option-deltas").locator("tbody tr", { hasText: "#5 Home" });
  await expect(responseRow.locator("td").nth(0)).toHaveText(facts.response.obs);
  await expect(responseRow.locator("td").nth(1)).toHaveText(facts.response.hyp);
  await guide.getByTestId("guide-compare").click();
  await expect(page.locator("#pc-title")).toBeFocused();

  // static defenders: the landing's static row
  await page.getByTestId("policy-select").selectOption("none");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 60_000 });
  await expect(page.getByTestId("policy-static-hint")).toBeVisible();
  await expect(responseRow.locator("td").nth(1)).toHaveText(facts.static.hyp);
  await expect(page.getByTestId("simulated-move")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("the response what-if link opens with the defender response preselected", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("featured-response-link").click();
  await expect(page).toHaveURL(/response=marker_follows/);
  await expect(page.getByTestId("policy-select")).toHaveValue("marker_follows");
  await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 60_000 });
  await expect(page.getByTestId("policy-comparison")).toHaveAttribute("data-verdict", "changes", { timeout: 60_000 });
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

test("back from the lab returns to the possession list, not the top of the landing", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("match-card-metrica-sample-2").click();
  const rows = page.getByTestId("possession-table").locator("tbody tr");
  await rows.nth(5).getByRole("link").click();
  await expect(page.getByTestId("pitch-canvas")).toBeVisible();
  await expect(page).toHaveTitle("Lab · FootyMind");
  await page.getByRole("button", { name: "Possessions" }).first().click();
  const heading = page.getByRole("heading", { name: "Possessions", level: 2 });
  await expect(heading).toBeFocused();
  await expect(heading).toBeInViewport();
  await expect(page.getByTestId("possessions-match")).toHaveText("Metrica Sports Sample Game 2");
  // the browser's Back button does the same
  await rows.nth(2).getByRole("link").click();
  await expect(page.getByTestId("pitch-canvas")).toBeVisible();
  await page.goBack();
  await expect(heading).toBeFocused();
  await expect(heading).toBeInViewport();
});

test("a direct #/match link shows that match's possessions, or why there are none", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto("/#/match/metrica-sample-2");
    await page.reload(); // a fresh load of the link, not an in-app navigation
    const heading = page.getByRole("heading", { name: "Possessions", level: 2 });
    await expect(heading).toBeInViewport();
    await expect(page.getByTestId("possessions-match")).toHaveText("Metrica Sports Sample Game 2");
    await page.goto("/#/match/no-such-match");
    await page.reload();
    await expect(page.getByTestId("unknown-match")).toBeInViewport();
  }
});

test("featured links are titled as the lab; an unknown match says so", async ({ page }) => {
  await page.goto("/#/at/metrica-sample-1?f=53708&tour=press");
  await expect(page.getByTestId("guide")).toBeVisible();
  await expect(page).toHaveTitle("Lab · FootyMind");

  await page.goto("/#/at/no-such-match?f=100");
  await expect(page.getByRole("heading", { level: 1, name: "Could not open this moment" })).toBeVisible();
  await expect(page.getByRole("alert")).toContainText('We could not find the match "no-such-match"');
  await page.getByRole("button", { name: "Browse possessions" }).click();
  await expect(page.getByTestId("unknown-match")).toContainText('This server has no match "no-such-match"');
});

test.describe("narrow screens", () => {
  test.use({ viewport: { width: 320, height: 640 } });
  test("a 320 px screen never scrolls sideways (landing, edit mode)", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("featured-card-response").getByTestId("featured-response")).toBeVisible({
      timeout: 60_000,
    });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
    await page.goto("/#/at/metrica-sample-2?f=31360&move=home_5,-29.04,9.47&response=learned_innovation");
    await expect(page.getByTestId("scenario-panel")).toHaveAttribute("data-status", "success", { timeout: 60_000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
  });
});

test.describe("tablet landscape", () => {
  test.use({ viewport: { width: 1180, height: 820 } });
  test("watching the tour keeps the play/pause control on screen", async ({ page }) => {
    await page.goto("/#/at/metrica-sample-1?f=53708&tour=press");
    await expect(page.getByTestId("guide")).toBeVisible();
    await page.getByTestId("guide-watch").click();
    await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Pause");
    await expect(page.getByTestId("play-toggle")).toBeInViewport({ ratio: 1 });
  });
});
