// Session 4 journey driver: the nine-step first-visit journey at three viewports (DPR 1).
// usage: WEB=<web dir> node journey.cjs <outDir> [baseURL]
// Env: ONLY=desktop,tablet,phone  FULL=1 (also full-page captures)
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
const FULL = !!process.env.FULL;
const log = [];

async function shot(page, name, full = false) {
  await page.mouse.move(1, 1);
  await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: full });
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, h: document.documentElement.scrollHeight }));
  log.push({ name, overflowX: Math.max(0, m.sw - m.cw), pageH: m.h });
}
async function settle(page) {
  await page.waitForFunction(() => {
    const p = document.querySelector('[data-testid="analysis-panel"]');
    return !p || p.getAttribute("aria-busy") === "false";
  }, null, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(300);
}

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
    await page.waitForSelector('[data-testid="possession-table"] tbody tr', { timeout: 30000 });
    await shot(page, `${vp}-01-arrive`);
    if (FULL) await shot(page, `${vp}-01-arrive-full`, true);

    // 2. pick featured -> open (the first featured link)
    const feat = page.locator('[data-testid="featured-link"]').first();
    await feat.scrollIntoViewIfNeeded();
    await shot(page, `${vp}-02-featured`);
    await feat.click();
    await page.waitForSelector('[data-testid="pitch-canvas"]');
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 }).catch(() => {});
    await page.waitForTimeout(500);
    await shot(page, `${vp}-03-featured-opened`);
    if (FULL) await shot(page, `${vp}-03-featured-opened-full`, true);

    // 3. watch observed possession: go to analyse mode, play 1.5 s
    const analyse = page.getByRole("radio", { name: /Replay/ });
    if (await analyse.count()) await analyse.first().click();
    await settle(page);
    await shot(page, `${vp}-04-observed-paused`);
    if (FULL) await shot(page, `${vp}-04-observed-paused-full`, true);
    await page.getByTestId("play-toggle").click();
    await page.waitForTimeout(1500);
    await shot(page, `${vp}-05-playing`);
    await page.getByTestId("play-toggle").click();
    await settle(page);

    // 4. why this moment
    const why = page.getByTestId("why-panel");
    if (await why.count()) {
      await why.scrollIntoViewIfNeeded();
      await shot(page, `${vp}-06-why`);
    }
    // 5. evidence: pass options table
    const po = page.getByTestId("pass-options");
    if (await po.count()) {
      await po.scrollIntoViewIfNeeded();
      await shot(page, `${vp}-07-evidence`);
    }
    // 6. similar moments
    const simBtn = page.getByTestId("similar-panel").getByRole("button").first();
    if (await simBtn.count()) {
      await simBtn.scrollIntoViewIfNeeded();
      await simBtn.click();
      await page.waitForTimeout(2500);
      await shot(page, `${vp}-08-similar`);
    }
    // player card
    await page.evaluate(() => window.scrollTo(0, 0));
    const menu = page.getByTestId("player-menu");
    const val = await menu.locator("option").nth(3).getAttribute("value");
    await menu.selectOption(val);
    await page.waitForTimeout(300);
    await shot(page, `${vp}-09-player-card`);

    // 7. move a defender: reopen featured scenario (edit mode)
    await page.goto(BASE + "/");
    await page.locator('[data-testid="featured-link"]').first().click();
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 }).catch(() => {});
    await page.waitForTimeout(400);
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, `${vp}-10-edit`);
    // 8. compare with response selector
    const sel = page.getByTestId("policy-select");
    await sel.scrollIntoViewIfNeeded();
    await sel.selectOption("marker_follows");
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await shot(page, `${vp}-11-response`);
    const cmp = page.getByTestId("policy-comparison");
    if (await cmp.count()) {
      await cmp.scrollIntoViewIfNeeded();
      await shot(page, `${vp}-12-comparison`);
    }
    // 9. what the model knows
    const ev = page.getByTestId("evidence");
    if (await ev.count()) {
      await ev.scrollIntoViewIfNeeded();
      await shot(page, `${vp}-13-caveats`);
    }
    if (FULL) await shot(page, `${vp}-13-full`, true);
    log.push({ vp, console: msgs });
    await ctx.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, "log.json"), JSON.stringify(log, null, 1));
  console.log(JSON.stringify(log, null, 1));
})();
