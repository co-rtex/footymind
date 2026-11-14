// axe-core over the session-4 journey states. usage: WEB=<web> AXE_JS=<axe.min.js> node axe4.cjs [base]
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const AXE = process.env.AXE_JS;
const BASE = process.argv[2] ?? "http://127.0.0.1:5190";
let total = 0;
async function run(page, name) {
  await page.addScriptTag({ path: AXE });
  const r = await page.evaluate(async () => {
    // eslint-disable-next-line no-undef
    const res = await axe.run(document, { resultTypes: ["violations"] });
    return res.violations.map((v) => ({ id: v.id, impact: v.impact, n: v.nodes.length, help: v.help, t: v.nodes.slice(0, 4).map((n) => n.target.join(" ") + " :: " + (n.any[0]?.message ?? "").slice(0, 120)) }));
  });
  total += r.length;
  console.log(`\n## ${name}: ${r.length} violation(s)`);
  for (const v of r) console.log(`- [${v.impact}] ${v.id} x${v.n}: ${v.help}\n    ${v.t.join("\n    ")}`);
}
const ready = (page) => page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const page = await b.newPage({ viewport: vp });
    const tag = vp.width > 1000 ? "desktop" : "phone";
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-card-broadcast"] [data-testid="featured-facts"]', { timeout: 30000 });
    await page.waitForFunction(() => !document.querySelector('[data-testid="featured-shadow-loading"]'), null, { timeout: 30000 });
    await run(page, `${tag} landing (live numbers loaded)`);
    await page.getByTestId("tour-link").click();
    await ready(page);
    await run(page, `${tag} tour step 1`);
    await page.getByTestId("guide-next").click();
    await ready(page);
    await page.getByTestId("pass-options-help").locator("summary").click();
    await page.getByTestId("legend-labels").locator("summary").click();
    await run(page, `${tag} tour step 2 + table help + labels key`);
    const menu = page.getByTestId("player-menu");
    await menu.selectOption(await menu.locator("option").nth(3).getAttribute("value"));
    await run(page, `${tag} player card`);
    await page.getByTestId("guide-next").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
    await page.getByTestId("policy-select").selectOption("marker_follows");
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
    await page.getByTestId("deltas-help").locator("summary").click();
    await run(page, `${tag} tour step 3 + response + delta help`);
    await page.close();
  }
  await b.close();
  console.log(`\nTOTAL violations: ${total}`);
})();
