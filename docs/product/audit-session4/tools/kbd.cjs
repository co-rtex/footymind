// Records the Tab order on the landing and in the guided tour (desktop).
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const BASE = process.argv[2] ?? "http://127.0.0.1:5190";
async function order(page, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    await page.keyboard.press("Tab");
    out.push(
      await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return "(body)";
        const label = el.getAttribute("aria-label") || el.dataset.testid || el.textContent.trim().replace(/\s+/g, " ").slice(0, 40);
        const vis = getComputedStyle(el).boxShadow !== "none" || getComputedStyle(el).outlineStyle !== "none";
        return `${el.tagName.toLowerCase()}: ${label}${vis ? "" : "  [NO VISIBLE FOCUS]"}`;
      }),
    );
  }
  return out;
}
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + "/");
  await page.waitForSelector('[data-testid="featured-facts"]', { timeout: 30000 });
  console.log("## landing\n" + (await order(page, 14)).map((s, i) => `${i + 1}. ${s}`).join("\n"));
  await page.goto(BASE + "/#/at/metrica-sample-1?f=53708&tour=press");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
  console.log("\n## lab, guided tour step 1\n" + (await order(page, 26)).map((s, i) => `${i + 1}. ${s}`).join("\n"));
  await b.close();
})();
