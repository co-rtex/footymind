// Similar-moments index missing for SkillCorner (scratch data dir); Metrica still has one.
const path = require("path");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.env.BASE ?? "http://127.0.0.1:5190";
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const page = await b.newPage({ viewport: vp });
    const errs = [];
    page.on("pageerror", (e) => errs.push(e.message));
    page.on("response", (r) => r.status() >= 400 && errs.push(`HTTP ${r.status()} ${r.url()}`));
    for (const [name, hash] of [
      ["skillcorner", "#/at/skillcorner-1899585?f=39177"],
      ["metrica", "#/at/metrica-sample-1?f=53708"],
    ]) {
      await page.goto(BASE + "/" + hash);
      await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 45000 });
      const find = page.getByTestId("similar-find");
      if (await find.count()) await find.click();
      await page.waitForFunction(() => ["success", "empty", "error"].includes(document.querySelector('[data-testid="similar-panel"]')?.getAttribute("data-status")), null, { timeout: 45000 });
      const p = page.getByTestId("similar-panel");
      await p.scrollIntoViewIfNeeded();
      console.log(`[${vp.width}] ${name}: status=${await p.getAttribute("data-status")} text=${((await p.innerText()) ?? "").replace(/\s+/g, " ").slice(0, 400)}`);
      await p.screenshot({ path: path.join(OUT, `noidx-${vp.width}-${name}.jpg`), type: "jpeg", quality: 70 });
    }
    console.log(`errs: ${JSON.stringify(errs)}`);
    await page.close();
  }
  await b.close();
})();
