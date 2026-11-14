// axe-core over the main screens. usage: node axe.cjs [base]
const path = require("path");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const AXE = process.env.AXE_JS; // axe.min.js installed outside the repository
const BASE = process.argv[2] ?? "http://127.0.0.1:5190";
let total = 0;
async function run(page, name) {
  await page.addScriptTag({ path: AXE });
  const r = await page.evaluate(async () => {
    // eslint-disable-next-line no-undef
    const res = await axe.run(document, { resultTypes: ["violations"] });
    return res.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      n: v.nodes.length,
      help: v.help,
      t: v.nodes.slice(0, 5).map((n) => n.target.join(" ") + " :: " + (n.any[0]?.message ?? n.all[0]?.message ?? "").slice(0, 160)),
    }));
  });
  total += r.length;
  console.log(`\n## ${name}: ${r.length} violation(s)`);
  for (const v of r) console.log(`- [${v.impact}] ${v.id} x${v.n}: ${v.help}\n    ${v.t.join("\n    ")}`);
}
const ready = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 45000 });
const waitSc = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 });
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 820, height: 1180 }, { width: 390, height: 844 }]) {
    const page = await b.newPage({ viewport: vp });
    const tag = `${vp.width}`;
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-card-response"] [data-testid="featured-response"]', { timeout: 60000 });
    await page.waitForFunction(() => !document.querySelector('[data-testid="featured-shadow-loading"]'), null, { timeout: 30000 });
    await run(page, `${tag} landing`);
    for (const fid of ["press", "broadcast", "response"]) {
      await page.goto(BASE + "/");
      await page.getByTestId(`tour-link-${fid}`).click();
      await ready(page);
      await page.getByTestId("guide-next").click();
      await ready(page);
      await page.getByTestId("similar-find").click().catch(() => {});
      await page
        .waitForFunction(() => document.querySelector('[data-testid="similar-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 45000 })
        .catch(() => {});
      await page.getByTestId("pass-options-help").locator("summary").click().catch(() => {});
      await page.getByTestId("legend-labels").locator("summary").click().catch(() => {});
      await run(page, `${tag} ${fid} step2 + similar + help`);
      await page.getByTestId("guide-next").click();
      await waitSc(page);
      await page.getByTestId("policy-select").selectOption("learned_innovation");
      await waitSc(page);
      await page.getByTestId("deltas-help").locator("summary").click().catch(() => {});
      await run(page, `${tag} ${fid} step3 learned_innovation`);
    }
    await page.goto(BASE + "/#/lab/metrica-sample-9%3Ap0001");
    await page.waitForSelector(".status--error, [role=alert]", { timeout: 30000 }).catch(() => {});
    await run(page, `${tag} unknown possession`);
    await page.close();
  }
  await b.close();
  console.log(`\nTOTAL violations: ${total}`);
})();
