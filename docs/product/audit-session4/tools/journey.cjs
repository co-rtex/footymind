// Session 4 journey driver (after): the nine-step first-visit journey through the guided tour,
// at three viewports, DPR 1. usage: WEB=<web dir> node journey2.cjs <outDir> [baseURL]
const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));

const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5190";
fs.mkdirSync(OUT, { recursive: true });
const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 834, height: 1112 },
  phone: { width: 390, height: 844 },
};
const log = [];
async function shot(page, name, full = false) {
  await page.mouse.move(1, 1);
  await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: full });
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, h: document.documentElement.scrollHeight }));
  log.push({ name, overflowX: Math.max(0, m.sw - m.cw), pageH: m.h });
}
const ok = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 }).catch(() => {});

(async () => {
  const browser = await chromium.launch();
  for (const [vp, size] of Object.entries(VIEWPORTS)) {
    if (process.env.ONLY && !process.env.ONLY.split(",").includes(vp)) continue;
    const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    const msgs = [];
    page.on("pageerror", (e) => msgs.push(`pageerror: ${e.message}`));
    page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && msgs.push(`${m.type()}: ${m.text()}`));

    // 1. arrive
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-card-broadcast"] [data-testid="featured-facts"]', { timeout: 30000 });
    await page.waitForFunction(() => !document.querySelector('[data-testid="featured-shadow-loading"]'), null, { timeout: 30000 }).catch(() => {});
    await shot(page, `${vp}-01-arrive`);
    await shot(page, `${vp}-01-arrive-full`, true);
    // 2. featured situation
    await page.getByTestId("featured-card-press").scrollIntoViewIfNeeded();
    await shot(page, `${vp}-02-featured`);

    // tour
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.getByTestId("tour-link").click();
    await page.waitForSelector('[data-testid="guide"]');
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    await page.waitForTimeout(300);
    await shot(page, `${vp}-03-tour-start`);
    // 3. watch
    await page.getByTestId("guide-watch").click();
    await page.waitForTimeout(1200);
    await shot(page, `${vp}-04-watch`);
    // 4. why
    await page.getByTestId("guide-next").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    await page.waitForTimeout(400);
    await shot(page, `${vp}-05-understand`);
    await page.getByTestId("guide").getByRole("button", { name: "Why this moment" }).click();
    await page.waitForTimeout(300);
    await shot(page, `${vp}-06-why`);
    // 5. evidence
    await page.getByTestId("guide").getByRole("button", { name: "Pass options" }).click();
    await page.getByTestId("pass-options-help").locator("summary").click();
    await page.waitForTimeout(300);
    await shot(page, `${vp}-07-evidence`);
    // 6. similar moments
    await page.getByTestId("guide").getByRole("button", { name: "Similar moments" }).click();
    await page.getByTestId("similar-find").click();
    await page.waitForSelector('[data-testid="similar-result"], [data-testid^="similar-empty"]', { timeout: 30000 }).catch(() => {});
    await page.locator("#similar-title").scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await shot(page, `${vp}-08-similar`);
    // player card
    const menu = page.getByTestId("player-menu");
    await menu.scrollIntoViewIfNeeded();
    await menu.selectOption(await menu.locator("option").nth(3).getAttribute("value"));
    await page.waitForTimeout(400);
    await shot(page, `${vp}-09-player-card`);
    await page.keyboard.press("Escape");
    // 7. move the defender
    await page.getByTestId("guide-next").scrollIntoViewIfNeeded();
    await page.getByTestId("guide-next").click();
    await ok(page);
    await page.waitForTimeout(500);
    await shot(page, `${vp}-10-what-if`);
    // 8. compare with the response selector
    await page.getByTestId("guide").getByRole("button", { name: "Defender response" }).click();
    await page.getByTestId("policy-select").selectOption("marker_follows");
    await ok(page);
    await page.waitForTimeout(800);
    await page.getByTestId("policy-select").scrollIntoViewIfNeeded();
    await shot(page, `${vp}-11-response`);
    const cmp = page.getByTestId("option-deltas");
    await cmp.scrollIntoViewIfNeeded();
    await shot(page, `${vp}-12-compare`);
    // 9. what the model knows
    await page.getByTestId("evidence").scrollIntoViewIfNeeded();
    await shot(page, `${vp}-13-caveats`);
    await page.getByTestId("side-nav").getByRole("button", { name: "Labels" }).click();
    await page.waitForTimeout(300);
    await shot(page, `${vp}-14-labels`);
    // broadcast: estimated players and ball
    await page.goto(BASE + "/#/at/skillcorner-1899585?f=39177&tour=broadcast");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    await page.waitForTimeout(400);
    await shot(page, `${vp}-15-broadcast-tour`);
    log.push({ vp, console: msgs });
    await ctx.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, "log.json"), JSON.stringify(log, null, 1));
  console.log(JSON.stringify(log.filter((e) => e.vp || e.overflowX), null, 1));
})();
