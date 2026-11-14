// Screenshots of the "Why this moment" and "Similar moments" panels on real data, 3 viewports.
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5183";
const VPS = [["desktop", { width: 1440, height: 900 }], ["tablet", { width: 834, height: 1112 }], ["phone", { width: 390, height: 844 }]];
(async () => {
  const b = await chromium.launch();
  const errors = [];
  for (const [vp, size] of VPS) {
    const page = await b.newPage({ viewport: size, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
    await page.waitForFunction(() => document.querySelector('[data-testid="why-panel"]')?.getAttribute("data-frame") === "53708");
    const why = page.getByTestId("why-panel");
    await why.scrollIntoViewIfNeeded();
    // trace: hover (desktop) or focus (keyboard) the first traceable statement
    const st = why.locator("button.why-text--trace").first();
    await st.focus();
    await page.waitForTimeout(100);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${OUT}/${vp}-20-why-trace.png` });
    await why.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-21-why-panel.png` });
    // step a few frames: the panel must keep its size
    const h0 = (await why.boundingBox()).height;
    const heights = [];
    for (let i = 0; i < 6; i++) {
      await page.locator("body").press("ArrowRight");
      await page.waitForTimeout(350);
      heights.push(Math.round((await why.boundingBox()).height - h0));
    }
    console.log(vp, "why height deltas over 6 frames:", JSON.stringify(heights));
    // similar moments
    await page.locator("body").press("ArrowLeft");
    await page.waitForTimeout(50);
    for (let i = 0; i < 5; i++) await page.locator("body").press("ArrowLeft");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "53708", null, { timeout: 15000 });
    const sim = page.getByTestId("similar-panel");
    await sim.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-22-similar-idle.png` });
    await page.getByTestId("similar-find").click();
    await page.getByTestId("similar-result").waitFor({ timeout: 30000 });
    await sim.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-23-similar-result.png` });
    await page.close();
  }
  // empty states: no decision near the frame (Metrica), no index (SkillCorner)
  const page = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  await page.goto(`${BASE}/#/lab/metrica-sample-1%3Ap0167?f=54500`);
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "54500");
  await page.getByTestId("similar-find").click();
  await page.getByTestId("similar-empty-no-decision").waitFor({ timeout: 30000 });
  await page.getByTestId("similar-panel").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${OUT}/desktop-25-similar-empty-no-decision.png` });
  const poss = await (await page.request.get(`${BASE}/api/matches/skillcorner-2017461/possessions?min_passes=3`)).json();
  await page.goto(`${BASE}/#/lab/${encodeURIComponent(poss[0].possession_id)}`);
  await page.getByTestId("why-panel").waitFor();
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false");
  await page.getByTestId("similar-find").click();
  await page.waitForFunction(() => document.querySelector('[data-testid="similar-panel"]')?.getAttribute("data-status") !== "loading");
  await page.getByTestId("similar-panel").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${OUT}/desktop-24-similar-empty-skillcorner.png` });
  console.log("skillcorner status:", await page.getByTestId("similar-panel").getAttribute("data-status"));
  console.log("errors:", JSON.stringify(errors));
  await b.close();
})();
