// Tables and clipped content at narrow widths.
const path = require("path");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const BASE = process.env.BASE ?? "http://127.0.0.1:5190";
const OUT = process.argv[2];
const measure = (page, where) =>
  page
    .evaluate(() => {
      const out = [];
      for (const el of document.querySelectorAll("table, .table-wrap, pre, .status-technical, select, .card, .feature-card, .policy-picker")) {
        const sw = el.scrollWidth;
        const cw = el.clientWidth;
        const ox = getComputedStyle(el).overflowX;
        if (sw > cw + 1) out.push(`${el.tagName.toLowerCase()}.${[...el.classList].join(".")}${el.dataset.testid ? "[" + el.dataset.testid + "]" : ""} sw=${sw} cw=${cw} overflowX=${ox}`);
      }
      return { out, doc: [document.documentElement.scrollWidth, innerWidth] };
    })
    .then((r) => console.log(`[${where}] doc=${r.doc.join("/")}\n   ${r.out.join("\n   ") || "(no clipped/scrolling elements)"}`));
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 390, height: 844 }, { width: 320, height: 640 }, { width: 820, height: 1180 }]) {
    const page = await b.newPage({ viewport: vp });
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-card-response"] [data-testid="featured-response"]', { timeout: 60000 });
    await page.waitForSelector('[data-testid="possession-table"]');
    await measure(page, `${vp.width} landing`);
    await page.screenshot({ path: path.join(OUT, `w${vp.width}-landing-full.jpg`), fullPage: true, type: "jpeg", quality: 50 });
    await page.goto(BASE + "/#/at/skillcorner-1899585?f=39177");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 45000 });
    await page.getByTestId("pass-options-help").locator("summary").click().catch(() => {});
    await page.getByTestId("similar-find").click().catch(() => {});
    await page.waitForTimeout(3000);
    await measure(page, `${vp.width} lab analyse`);
    await page.screenshot({ path: path.join(OUT, `w${vp.width}-lab-full.jpg`), fullPage: true, type: "jpeg", quality: 50 });
    await page.goto(BASE + "/#/at/metrica-sample-2?f=31360&move=home_5,-29.04,9.47&response=learned_innovation");
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 });
    await page.waitForTimeout(1500);
    await measure(page, `${vp.width} lab edit`);
    await page.screenshot({ path: path.join(OUT, `w${vp.width}-edit-full.jpg`), fullPage: true, type: "jpeg", quality: 50 });
    await page.close();
  }
  await b.close();
})();
