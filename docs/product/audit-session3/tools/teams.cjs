// Screenshots: team names (SkillCorner), Metrica anonymised, Similar panel validated with clocks.
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5183";
(async () => {
  const b = await chromium.launch();
  const errors = [];
  for (const [vp, size] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]]) {
    const page = await b.newPage({ viewport: size, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(BASE + "/");
    await page.getByTestId("possession-table").locator("tbody tr").first().waitFor();
    await page.screenshot({ path: `${OUT}/${vp}-30-landing-featured.png` });
    await page.getByTestId("match-card-skillcorner-1874553").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-31-match-cards.png` });
    await page.getByTestId("match-card-skillcorner-1874553").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="possessions-match"]')?.textContent?.includes("Brisbane"));
    await page.getByTestId("possessions-match").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-32-possessions-named.png` });
    await page.getByTestId("possession-table").locator("tbody tr").first().getByRole("link").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    await page.mouse.move(0, 0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${OUT}/${vp}-33-lab-named.png` });
    await page.getByRole("group", { name: "Legend" }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-34-legend-named.png` });
    await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "53708");
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${OUT}/${vp}-35-lab-metrica-anon.png` });
    await page.getByTestId("similar-find").click();
    await page.getByTestId("similar-result").waitFor({ timeout: 30000 });
    await page.waitForTimeout(300);
    await page.getByTestId("similar-panel").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-36-similar-validated.png` });
    // SkillCorner similar moments with named neighbours
    await page.goto(BASE + "/#/at/skillcorner-1899585?f=39177");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "39177", null, { timeout: 30000 });
    await page.getByTestId("similar-find").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="similar-panel"]')?.getAttribute("data-status") !== "loading", null, { timeout: 60000 });
    await page.waitForTimeout(300);
    await page.getByTestId("similar-panel").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-37-similar-skillcorner.png` });
    console.log(vp, "sc similar status", await page.getByTestId("similar-panel").getAttribute("data-status"));
    await page.close();
  }
  console.log("errors", JSON.stringify(errors));
  await b.close();
})();
