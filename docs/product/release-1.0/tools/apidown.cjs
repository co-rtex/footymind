// API stopped mid-session. Kills the QA API (port 8790) after the lab has loaded.
const path = require("path");
const { execSync } = require("child_process");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.env.BASE ?? "http://127.0.0.1:5190";
const t = (page) =>
  page.evaluate(() =>
    [...document.querySelectorAll(".status")].map((s) => `${s.getAttribute("role")}: ${s.innerText.replace(/\s+/g, " ").slice(0, 200)}`),
  );
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(BASE + "/#/at/metrica-sample-1?f=53708&tour=press");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 45000 });
  console.log("lab loaded; stopping API");
  execSync("pkill -f 'footymind serve --port [8]790' || true");
  await page.waitForTimeout(1500);
  // step a frame -> new analysis request
  await page.locator("body").click({ position: { x: 5, y: 300 } });
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(4000);
  console.log("after step:", JSON.stringify(await t(page)));
  await page.screenshot({ path: path.join(OUT, "apidown-01-step.jpg"), type: "jpeg", quality: 65 });
  // similar
  await page.getByTestId("similar-find").click().catch(() => console.log("no similar-find"));
  await page.waitForTimeout(3000);
  console.log("after similar:", JSON.stringify(await t(page)));
  // edit scenario
  await page.getByTestId("guide-next").click();
  await page.getByTestId("guide-next").click();
  await page.waitForTimeout(4000);
  console.log("after what-if:", JSON.stringify(await t(page)));
  await page.screenshot({ path: path.join(OUT, "apidown-02-whatif.jpg"), type: "jpeg", quality: 65 });
  // back to landing
  await page.goto(BASE + "/");
  await page.waitForTimeout(4000);
  console.log("landing:", JSON.stringify(await t(page)));
  await page.screenshot({ path: path.join(OUT, "apidown-03-landing.jpg"), type: "jpeg", quality: 65 });
  await b.close();
})();
