// QA journey: landing -> tour for each featured situation -> watch/pause -> why -> pass options
// -> similar -> move -> compare views -> 7 policies -> limitations -> back -> another analysis.
// usage: node journey.cjs <outdir> [viewport tags...]
const path = require("path");
const fs = require("fs");
const WEB = process.env.WEB; // path to the repository's web/ directory
const { chromium } = require(path.join(WEB, "node_modules/playwright"));
const BASE = process.env.BASE ?? "http://127.0.0.1:5190";
const OUT = process.argv[2] ?? "/tmp/qa-out";
const VPS = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 820, height: 1180 },
  tabletL: { width: 1180, height: 820 },
  phone: { width: 390, height: 844 },
};
const tags = process.argv.slice(3).length ? process.argv.slice(3) : Object.keys(VPS);
fs.mkdirSync(OUT, { recursive: true });
const log = [];
const note = (s) => {
  log.push(s);
  console.log(s);
};

async function overflow(page, where) {
  const r = await page.evaluate(() => {
    const iw = window.innerWidth;
    const sw = document.documentElement.scrollWidth;
    const offenders = [];
    if (sw > iw) {
      for (const el of document.querySelectorAll("body *")) {
        const b = el.getBoundingClientRect();
        if (b.right > iw + 1 && b.width > 0) {
          // skip descendants of scroll containers
          let p = el.parentElement;
          let clipped = false;
          while (p) {
            const s = getComputedStyle(p);
            if (/(auto|scroll|hidden)/.test(s.overflowX)) {
              clipped = true;
              break;
            }
            p = p.parentElement;
          }
          if (!clipped) offenders.push(`${el.tagName.toLowerCase()}.${[...el.classList].join(".")} r=${Math.round(b.right)}`);
        }
      }
    }
    return { iw, sw, offenders: offenders.slice(0, 6) };
  });
  if (r.sw > r.iw) note(`  OVERFLOW @${where}: scrollWidth ${r.sw} > ${r.iw}: ${r.offenders.join(" | ")}`);
  return r;
}

async function smallTargets(page, where) {
  const r = await page.evaluate(() => {
    const out = [];
    for (const el of document.querySelectorAll("button, a[href], select, input, summary, [role=radio], [role=tab]")) {
      const b = el.getBoundingClientRect();
      const st = getComputedStyle(el);
      if (b.width === 0 || b.height === 0 || st.visibility === "hidden") continue;
      if (el.closest(".sr-only") || el.classList.contains("skip-link")) continue;
      // checkbox inside a label: measure the label
      let t = b;
      if (el.tagName === "INPUT" && el.closest("label")) t = el.closest("label").getBoundingClientRect();
      // inline text links are exempt (WCAG 2.5.8 inline exception)
      if (el.tagName === "A" && getComputedStyle(el).display === "inline" && el.closest("p,li,dd,td")) continue;
      if (t.height < 24 || t.width < 24) out.push(`${el.tagName.toLowerCase()}${el.dataset.testid ? "[" + el.dataset.testid + "]" : ""}"${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 30)}" ${Math.round(t.width)}x${Math.round(t.height)}`);
    }
    return out;
  });
  if (r.length) note(`  SMALL TARGETS (<24px) @${where}: ${r.length}: ${r.slice(0, 8).join(" | ")}`);
}

const waitAn = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false", null, { timeout: 45000 });
const waitSc = (page) =>
  page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 60000 });

(async () => {
  const browser = await chromium.launch();
  for (const tag of tags) {
    const vp = VPS[tag];
    const ctx = await browser.newContext({ viewport: vp, hasTouch: tag === "phone", isMobile: tag === "phone" });
    const page = await ctx.newPage();
    const errs = [];
    page.on("pageerror", (e) => errs.push(`pageerror: ${e.message}`));
    page.on("console", (m) => m.type() === "error" && errs.push(`console: ${m.text()}`));
    page.on("response", (r) => r.status() >= 400 && errs.push(`HTTP ${r.status()} ${r.url()}`));
    page.on("requestfailed", (r) => {
      const f = r.failure()?.errorText ?? "";
      if (!/ERR_ABORTED/.test(f)) errs.push(`reqfailed ${f} ${r.url()}`);
    });
    let n = 0;
    const shot = async (name, full = false) => {
      n++;
      const f = `${tag}-${String(n).padStart(2, "0")}-${name}.jpg`;
      await page.screenshot({ path: path.join(OUT, f), fullPage: full, type: "jpeg", quality: 70 });
      await overflow(page, name);
      return f;
    };
    note(`\n=== ${tag} ${vp.width}x${vp.height}`);
    const t0 = Date.now();
    await page.goto(BASE + "/");
    await page.waitForSelector('[data-testid="featured-card-broadcast"] [data-testid="featured-facts"]', { timeout: 60000 });
    await page.waitForSelector('[data-testid="featured-card-response"] [data-testid="featured-response"]', { timeout: 60000 });
    note(`  landing live numbers in ${Date.now() - t0} ms`);
    await shot("landing");
    await shot("landing-full", true);
    await smallTargets(page, "landing");

    for (const fid of ["press", "broadcast", "response"]) {
      await page.goto(BASE + "/");
      const link = page.getByTestId(`tour-link-${fid}`);
      await link.waitFor({ timeout: 30000 }).catch(() => {});
      if (!(await link.count())) {
        note(`  no tour link for ${fid}`);
        continue;
      }
      await link.click();
      await page.waitForSelector('[data-testid="guide"]', { timeout: 30000 });
      await waitAn(page);
      const url = page.url();
      const mt = await page.getByTestId("match-title").textContent();
      const af = await page.getByTestId("analysis-panel").getAttribute("data-frame");
      note(`  [${fid}] url=${url.replace(BASE, "")} match="${mt}" frame=${af}`);
      await shot(`${fid}-tour-start`);
      await smallTargets(page, `${fid}-lab`);
      // watch
      await page.getByTestId("guide-watch").click();
      await page.waitForTimeout(1500);
      await shot(`${fid}-watching`);
      // pause via play toggle
      await page.getByTestId("play-toggle").click();
      await page.waitForTimeout(300);
      const lab = await page.getByTestId("play-toggle").getAttribute("aria-label");
      note(`  [${fid}] after pause toggle label=${lab}`);
      await waitAn(page).catch(() => note(`  [${fid}] analysis did not settle after pause`));
      await shot(`${fid}-paused`);
      // step 2
      await page.getByTestId("guide-next").click();
      await waitAn(page);
      const af2 = await page.getByTestId("analysis-panel").getAttribute("data-frame");
      const wf = await page.getByTestId("why-panel").getAttribute("data-frame");
      const nWhy = await page.getByTestId("why-statement").count();
      const nOpt = await page.getByTestId("pass-options").locator("tbody tr").count().catch(() => -1);
      note(`  [${fid}] step2 frame=${af2} why-frame=${wf} statements=${nWhy} passOptions=${nOpt}`);
      await shot(`${fid}-understand`);
      // why statement focus/trace
      if (nWhy) {
        await page.getByTestId("why-statement").first().hover().catch(() => {});
      }
      await page.getByTestId("guide").getByRole("button", { name: "Similar moments" }).click();
      const find = page.getByTestId("similar-find");
      if (await find.count()) await find.click();
      await page
        .waitForFunction(() => ["success", "empty", "error"].includes(document.querySelector('[data-testid="similar-panel"]')?.getAttribute("data-status")), null, { timeout: 45000 })
        .catch(() => {});
      const ss = await page.getByTestId("similar-panel").getAttribute("data-status");
      const ni = await page.getByTestId("similar-item").count();
      note(`  [${fid}] similar status=${ss} items=${ni}`);
      await shot(`${fid}-similar`);
      // step 3
      await page.getByTestId("guide-next").click();
      await waitSc(page);
      const head = (await page.getByTestId("scenario-headline").textContent().catch(() => "")) ?? "";
      note(`  [${fid}] step3 headline="${head.slice(0, 140)}" policy=${await page.getByTestId("policy-select").inputValue()}`);
      await shot(`${fid}-what-if`);
      for (const v of ["Observed", "Hypothetical", "Difference"]) {
        await page.getByTestId("scenario-view").getByRole("radio", { name: v }).click();
        await page.waitForTimeout(200);
      }
      // all policies
      const opts = await page.getByTestId("policy-select").locator("option").evaluateAll((os) => os.map((o) => o.value));
      note(`  [${fid}] policies(${opts.length})=${opts.join(",")}`);
      for (const o of opts) {
        await page.getByTestId("policy-select").selectOption(o);
        const ok = await waitSc(page).then(
          () => true,
          () => false,
        );
        const st = await page.getByTestId("scenario-panel").getAttribute("data-status");
        const sim = await page.getByTestId("simulated-move").count();
        const noSim = await page.getByTestId("no-simulated-moves").count();
        const cmp = await page.getByTestId("policy-comparison").count();
        note(`    policy ${o}: ok=${ok} status=${st} simulatedMoves=${sim} noSimNote=${noSim} comparison=${cmp}`);
        if (fid === "response" || (fid === "press" && tag === "desktop")) await shot(`${fid}-policy-${o}`);
      }
      // limitations: evidence card
      const ev = page.getByTestId("evidence");
      if (await ev.count()) {
        await ev.scrollIntoViewIfNeeded();
        await shot(`${fid}-evidence`);
      }
      await page.getByTestId("guide-finish").click().catch(() => note(`  [${fid}] no guide-finish`));
    }

    // move a defender by drag (press situation, edit mode) using move form via keyboard path
    await page.goto(BASE + "/#/at/metrica-sample-1?f=53708");
    await waitAn(page);
    await page.getByTestId("mode").getByRole("radio", { name: "Edit scenario" }).click();
    await page.getByTestId("player-menu").selectOption("home_9");
    await page.getByTestId("move-x").fill("48");
    await page.getByTestId("move-y").fill("-8");
    await page.getByTestId("move-submit").click();
    await waitSc(page).catch(() => note("  move-form scenario did not settle"));
    await shot("move-form-result");

    // back to match selection, another analysis
    await page.getByRole("button", { name: /Possessions/ }).first().click();
    await page.waitForSelector('[data-testid="possession-table"]', { timeout: 30000 });
    await page.getByTestId("match-card-skillcorner-2017461").click();
    await page.waitForFunction(() => document.querySelector('[data-testid="possessions-match"]')?.textContent?.length > 0);
    await page.waitForSelector('[data-testid="possession-table"] tbody tr', { timeout: 30000 });
    const pm = await page.getByTestId("possessions-match").textContent();
    note(`  back: possessions for "${pm}"`);
    await shot("back-match-select");
    await page.getByTestId("possession-table").locator("tbody tr").nth(2).getByRole("link").click();
    await page.waitForSelector('[data-testid="pitch-canvas"]', { timeout: 30000 });
    await page.getByTestId("play-toggle").click();
    await page.waitForTimeout(1200);
    await page.getByTestId("play-toggle").click();
    await waitAn(page);
    note(`  another analysis: match="${await page.getByTestId("match-title").textContent()}" frame=${await page.getByTestId("analysis-panel").getAttribute("data-frame")}`);
    await shot("another-analysis");
    // pitch size
    const pb = await page.getByTestId("pitch-canvas").boundingBox();
    note(`  pitch canvas ${Math.round(pb.width)}x${Math.round(pb.height)}`);

    note(`  ERRORS (${errs.length}):\n    ${[...new Set(errs)].join("\n    ")}`);
    await ctx.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, "journey.log"), log.join("\n"));
})().catch((e) => {
  console.error("FATAL", e);
  fs.writeFileSync(path.join(OUT, "journey.log"), log.join("\n") + "\nFATAL " + e.stack);
  process.exit(1);
});
