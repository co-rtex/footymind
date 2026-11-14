// Runs axe-core on key states; prints violations (id, impact, count, sample targets).
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const AXE = process.env.AXE_JS ?? path.join(__dirname, "axe/node_modules/axe-core/axe.min.js");
const BASE = process.argv[2] ?? "http://127.0.0.1:5183";
async function run(page, name) {
  await page.addScriptTag({ path: AXE });
  const r = await page.evaluate(async () => {
    // eslint-disable-next-line no-undef
    const res = await axe.run(document, { resultTypes: ["violations"] });
    return res.violations.map((v) => ({ id: v.id, impact: v.impact, n: v.nodes.length, help: v.help, t: v.nodes.slice(0, 4).map((n) => n.target.join(" ") + " :: " + (n.any[0]?.message ?? "").slice(0, 110)) }));
  });
  console.log(`\n## ${name}`);
  for (const v of r) console.log(`- [${v.impact}] ${v.id} x${v.n}: ${v.help}\n    ${v.t.join("\n    ")}`);
}
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + "/");
  await page.getByTestId("possession-table").locator("tbody tr").first().waitFor();
  await run(page, "landing");
  await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false");
  await page.getByRole("checkbox", { name: /Defensive shadow/ }).check();
  await page.waitForFunction(() => document.querySelector('[data-testid="shadow-panel"]')?.getAttribute("data-frame") === "53708", null, { timeout: 30000 });
  await run(page, "lab analyse + shadow");
  await page.goto(BASE + "/");
  await page.getByTestId("featured-link").click();
  await page.getByTestId("option-deltas").waitFor({ timeout: 30000 });
  await run(page, "scenario");
  await b.close();
})();
