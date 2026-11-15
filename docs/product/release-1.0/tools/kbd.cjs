// Keyboard-only journey: landing -> tour -> watch/pause -> understand -> what-if -> policies -> back.
const path = require("path");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const BASE = process.env.BASE ?? "http://127.0.0.1:5190";
const OUT = process.argv[2];
const desc = (page) =>
  page.evaluate(() => {
    const el = document.activeElement;
    if (!el || el === document.body) return { s: "(body)", vis: true };
    const label = (el.getAttribute("aria-label") || el.dataset.testid || el.textContent.trim().replace(/\s+/g, " ")).slice(0, 50);
    const cs = getComputedStyle(el);
    const vis = cs.boxShadow !== "none" || (cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0);
    const r = el.getBoundingClientRect();
    const onScreen = r.bottom > 0 && r.top < innerHeight && r.width > 0;
    return { s: `${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}: ${label}`, vis, onScreen };
  });
async function tabTo(page, pred, max = 120, logAll = false) {
  const seen = [];
  for (let i = 0; i < max; i++) {
    await page.keyboard.press("Tab");
    const d = await desc(page);
    seen.push(d);
    if (logAll || !d.vis || d.onScreen === false) console.log(`   ${i + 1}. ${d.s}${d.vis ? "" : " [NO VISIBLE FOCUS]"}${d.onScreen === false ? " [OFFSCREEN]" : ""}`);
    if (pred(d.s)) return seen;
  }
  throw new Error("not reached: " + seen.slice(-5).map((d) => d.s).join(" | "));
}
const st = (page) => page.evaluate(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status"));
(async () => {
  const b = await chromium.launch();
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    console.log(`\n===== ${vp.width}`);
    const page = await b.newPage({ viewport: vp });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-facts"]', { timeout: 30000 });
    console.log("## landing -> tour link");
    await tabTo(page, (s) => /^a: tour-link(-hero)?$/.test(s), 40, true);
    await page.keyboard.press("Enter");
    await page.waitForSelector('[data-testid="guide"]', { timeout: 30000 });
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    console.log("  focus after route change:", (await desc(page)).s);
    console.log("## lab: Tab to Play the possession");
    await tabTo(page, (s) => /guide-watch|Play the possession/.test(s), 60, true);
    await page.keyboard.press("Enter");
    await page.waitForTimeout(800);
    console.log("  playing:", await page.getByTestId("play-toggle").getAttribute("aria-label"));
    console.log("## Tab to play toggle and pause");
    await tabTo(page, (s) => /^button: Pause/.test(s), 60);
    await page.keyboard.press("Enter");
    await page.waitForTimeout(300);
    console.log("  after Enter:", await page.getByTestId("play-toggle").getAttribute("aria-label"));
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    // arrow keys on the scrubber
    await tabTo(page, (s) => /scrubber|slider|input/.test(s), 10, true);
    await page.keyboard.press("ArrowRight");
    await page.screenshot({ path: path.join(OUT, `kbd-${vp.width}-scrubber.jpg`), type: "jpeg", quality: 60 });
    console.log("## guide step 2 via keyboard: go back up with Shift+Tab to guide-next");
    await page.getByTestId("guide-next").focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 30000 });
    console.log("  focus after step change:", (await desc(page)).s);
    await tabTo(page, (s) => /Similar moments/.test(s), 20, true);
    await page.keyboard.press("Enter");
    console.log("  focus after jump:", (await desc(page)).s);
    await tabTo(page, (s) => /similar-find|Find similar|^a: Open|Open/.test(s), 20, true);
    console.log("## Tab through why statements");
    await page.locator("#why-title").focus().catch(() => {});
    await tabTo(page, (s) => /why-statement|^li/.test(s), 15, true).catch((e) => console.log("  ", e.message));
    console.log("## step 3");
    await page.getByTestId("guide-next").focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 });
    console.log("  focus after step 3:", (await desc(page)).s);
    console.log("## Tab to policy select, change with keyboard");
    await tabTo(page, (s) => /policy-select/.test(s), 80);
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("ArrowDown");
      await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 });
      console.log(`   policy=${await page.getByTestId("policy-select").inputValue()} status=${await st(page)} focus=${(await desc(page)).s.slice(0, 40)}`);
    }
    await page.screenshot({ path: path.join(OUT, `kbd-${vp.width}-policy.jpg`), type: "jpeg", quality: 60 });
    console.log("## keyboard move via player menu + move form");
    await page.getByTestId("player-menu").focus();
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowDown");
    await page.waitForTimeout(300);
    console.log("  focus after menu pick:", (await desc(page)).s);
    console.log("## Escape and back to Possessions");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: /Possessions/ }).first().focus();
    await page.keyboard.press("Enter");
    await page.waitForSelector('[data-testid="possession-table"]', { timeout: 30000 });
    console.log("  focus after back:", (await desc(page)).s, " scrollY=", await page.evaluate(() => scrollY));
    await page.close();
  }
  await b.close();
})().catch((e) => {
  console.error("FATAL", e.message);
  process.exit(1);
});
