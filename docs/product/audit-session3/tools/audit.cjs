// Product audit driver: screenshots every key state at three viewports (DPR 1).
// usage: node audit.cjs <outDir> [baseURL]
const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));

const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5183";
fs.mkdirSync(OUT, { recursive: true });

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 834, height: 1112 },
  phone: { width: 390, height: 844 },
};
const MARGIN = 3.5;

async function toScreen(page, x, y) {
  const box = await page.getByTestId("pitch-canvas").boundingBox();
  const s = Math.min(box.width / (105 + 2 * MARGIN), box.height / (68 + 2 * MARGIN));
  return { x: box.x + box.width / 2 + x * s, y: box.y + box.height / 2 - y * s };
}
async function waitAnalysis(page) {
  const p = page.getByTestId("analysis-panel");
  await p.waitFor();
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
  return Number(await p.getAttribute("data-frame"));
}

const log = [];
async function shot(page, name, full = false) {
  const f = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: f, fullPage: full });
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
  log.push({ name, overflowX: m.sw > m.cw ? m.sw - m.cw : 0 });
}

(async () => {
  const browser = await chromium.launch();
  for (const [vp, size] of Object.entries(VIEWPORTS)) {
    if (process.env.ONLY && !process.env.ONLY.split(",").includes(vp)) continue;
    const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    const errs = [];
    page.on("pageerror", (e) => errs.push(e.message));
    page.on("console", (m) => m.type() === "error" && errs.push(m.text()));

    await page.goto(BASE + "/");
    await page.getByTestId("possession-table").locator("tbody tr").first().waitFor();
    await shot(page, `${vp}-01-landing`);
    await shot(page, `${vp}-01-landing-full`, true);

    // match selection: pick a SkillCorner match
    const cards = page.locator(".match-card");
    await cards.nth(4).click();
    await page.getByTestId("possession-table").locator("tbody tr").first().waitFor();
    await page.locator("#match-title").scrollIntoViewIfNeeded();
    await shot(page, `${vp}-02-match-selected`);

    // lab: featured metrica frame (pid route)
    await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
    const frame = await waitAnalysis(page);
    await page.mouse.move(0, 0);
    await shot(page, `${vp}-03-lab-paused`);
    await shot(page, `${vp}-03-lab-paused-full`, true);

    // player inspection: click the carrier
    const a = await (await page.request.get(`${BASE}/api/matches/metrica-sample-1/frames/${frame}/analysis?grid=false`)).json();
    const carrier = a.players.find((p) => p.player_id === a.carrier_id) ?? a.players.find((p) => p.x !== null);
    let pt = await toScreen(page, carrier.x, carrier.y);
    await page.mouse.click(pt.x, pt.y);
    await page.waitForTimeout(200);
    await shot(page, `${vp}-04-player-inspector`);
    await shot(page, `${vp}-04-player-inspector-full`, true);

    // passing lanes highlight: hover first pass option row
    const row = page.getByTestId("pass-options").locator("tbody tr").first();
    if (await row.count()) {
      await row.scrollIntoViewIfNeeded();
      await row.hover();
      await page.waitForTimeout(150);
      await shot(page, `${vp}-05-pass-lane-highlight`);
    }

    // shadow layer
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.getByRole("checkbox", { name: /Defensive shadow/ }).check();
    await page.getByTestId("shadow-panel").waitFor();
    await page.waitForFunction(() => document.querySelector('[data-testid="shadow-panel"]')?.getAttribute("data-frame") === "53708", null, { timeout: 30000 });
    await page.mouse.move(0, 0);
    await shot(page, `${vp}-06-shadow`);
    await shot(page, `${vp}-06-shadow-full`, true);
    await page.getByRole("checkbox", { name: /Defensive shadow/ }).uncheck();

    // edit: drag the nearest defender 3 m towards the ball
    await page.getByRole("radio", { name: "Edit scenario" }).click();
    await page.getByTestId("scenario-panel").waitFor();
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, `${vp}-07-edit-empty`);
    const def = a.players
      .filter((p) => p.team !== a.attacking_team && p.x !== null && !p.is_gk && p.reach_radius_m !== null)
      .sort((p, q) => Math.hypot(p.x - a.ball_x, p.y - a.ball_y) - Math.hypot(q.x - a.ball_x, q.y - a.ball_y))[0];
    const len = Math.hypot(a.ball_x - def.x, a.ball_y - def.y) || 1;
    const to = { x: def.x + ((a.ball_x - def.x) / len) * 3, y: def.y + ((a.ball_y - def.y) / len) * 3 };
    const s = await toScreen(page, def.x, def.y);
    const e = await toScreen(page, to.x, to.y);
    await page.mouse.move(s.x, s.y);
    await page.mouse.down();
    await page.mouse.move((s.x + e.x) / 2, (s.y + e.y) / 2, { steps: 5 });
    await shot(page, `${vp}-08-dragging`);
    await page.mouse.move(e.x, e.y, { steps: 5 });
    await page.mouse.up();
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
    await page.mouse.move(0, 0);
    await shot(page, `${vp}-09-scenario`);
    await shot(page, `${vp}-09-scenario-full`, true);
    await page.getByRole("radio", { name: "Difference" }).click();
    await page.waitForTimeout(150);
    await shot(page, `${vp}-10-difference`);
    // marker follows: move an attacker instead
    const policy = page.getByRole("radio", { name: "Marker follows" });
    const sel = page.getByTestId("policy-select");
    if (await sel.count()) {
      await sel.selectOption("marker_follows");
      await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
      await page.getByTestId("policy-comparison").waitFor({ timeout: 30000 }).catch(() => {});
      await page.getByTestId("policy-picker").scrollIntoViewIfNeeded();
      await shot(page, `${vp}-10b-policy`);
      await page.getByTestId("policy-comparison").scrollIntoViewIfNeeded().catch(() => {});
      await shot(page, `${vp}-10c-policy-comparison`);
    } else if (await policy.count()) {
      await policy.click();
      await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
    }
    // evidence expanded
    const det = page.getByTestId("evidence").locator("details summary");
    if (await det.count()) {
      await det.first().click();
      await page.getByTestId("evidence").scrollIntoViewIfNeeded();
      await shot(page, `${vp}-11-evidence`);
    }

    // featured deep link
    await page.goto(BASE + "/");
    await page.getByTestId("featured-link").click();
    await page.getByTestId("option-deltas").waitFor({ timeout: 30000 });
    await page.mouse.move(0, 0);
    await shot(page, `${vp}-12-featured`);

    // error states
    await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap9999");
    await page.waitForTimeout(1500);
    await shot(page, `${vp}-13-bad-possession`);
    await page.goto(BASE + "/#/lab/garbage");
    await page.waitForTimeout(1500);
    await shot(page, `${vp}-13b-bad-route`);
    await page.goto(BASE + "/#/at/metrica-sample-1?f=1");
    await page.waitForTimeout(1500);
    await shot(page, `${vp}-13c-at-no-possession`);
    log.push({ vp, errors: errs });
    await ctx.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, "log.json"), JSON.stringify(log, null, 1));
  console.log(JSON.stringify(log));
})().catch((e) => {
  console.error(e);
  fs.writeFileSync(path.join(OUT, "log.json"), JSON.stringify(log, null, 1));
  process.exit(1);
});
