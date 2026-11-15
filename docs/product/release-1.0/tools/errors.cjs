// Error / empty / loading states. usage: node errors.cjs <outdir> [base]
const path = require("path");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5190";
const CASES = [
  ["unknown-match-at", "#/at/no-such-match?f=100"],
  ["unknown-match-home", "#/match/no-such-match"],
  ["unknown-possession", "#/lab/metrica-sample-1%3Ap9999"],
  ["frame-out-of-range", "#/at/metrica-sample-1?f=99999999"],
  ["bogus-policy", "#/at/metrica-sample-2?f=31360&move=home_5,-29.04,9.47&response=bogus_policy"],
  ["bogus-tour", "#/at/metrica-sample-1?f=53708&tour=nope"],
  ["bogus-move-player", "#/at/metrica-sample-1?f=53708&move=home_99,10,10"],
];
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const page = await b.newPage({ viewport: vp });
    for (const [name, hash] of CASES) {
      const errs = [];
      const onE = (e) => errs.push(`pageerror ${e.message}`);
      const onC = (m) => m.type() === "error" && errs.push(`console ${m.text().slice(0, 140)}`);
      page.on("pageerror", onE);
      page.on("console", onC);
      await page.goto("about:blank");
      await page.goto(BASE + "/" + hash);
      await page.waitForTimeout(5000);
      const txt = await page.evaluate(() => {
        const st = [...document.querySelectorAll(".status")].map((s) => s.innerText.replace(/\s+/g, " ").slice(0, 300));
        const h1 = [...document.querySelectorAll("h1")].map((h) => h.innerText.replace(/\s+/g, " "));
        const sp = document.querySelector('[data-testid="scenario-panel"]');
        return { st, h1, title: document.title, sc: sp ? sp.innerText.replace(/\s+/g, " ").slice(0, 400) : null, focus: document.activeElement?.tagName + "." + document.activeElement?.className };
      });
      console.log(`\n[${vp.width}] ${name} ${hash}\n  h1=${JSON.stringify(txt.h1)} title=${txt.title} focus=${txt.focus}\n  status=${JSON.stringify(txt.st)}\n  scenario=${txt.sc}\n  errs=${JSON.stringify(errs)}`);
      await page.screenshot({ path: path.join(OUT, `err-${vp.width}-${name}.jpg`), type: "jpeg", quality: 65 });
      page.off("pageerror", onE);
      page.off("console", onC);
    }
    await page.close();
  }
  await b.close();
})();
