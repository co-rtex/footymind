// Tab order through the lab in edit mode; records each focused element.
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const BASE = process.argv[2] ?? "http://127.0.0.1:5183";
const OUT = process.argv[3];
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false");
  await page.getByRole("radio", { name: "Edit scenario" }).click();
  await page.locator("body").click({ position: { x: 5, y: 300 } });
  const seen = [];
  for (let i = 0; i < 40; i++) {
    await page.keyboard.press("Tab");
    const d = await page.evaluate(() => {
      const e = document.activeElement;
      if (!e) return "none";
      const name = e.getAttribute("aria-label") || e.textContent?.trim().slice(0, 40) || "";
      return `${e.tagName.toLowerCase()}${e.getAttribute("role") ? "[" + e.getAttribute("role") + "]" : ""} "${name}"`;
    });
    seen.push(d);
    if (OUT && i === 12) await page.screenshot({ path: OUT });
  }
  console.log(seen.map((s, i) => `${i + 1}. ${s}`).join("\n"));
  await b.close();
})();
