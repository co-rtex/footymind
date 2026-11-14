// Measures scrub cost: per input event, time until the next painted frame, and main-thread work.
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const BASE = process.argv[2] ?? "http://127.0.0.1:5183";
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  let analysisRequests = 0;
  page.on("request", (r) => r.url().includes("/analysis") && analysisRequests++);
  await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("aria-busy") === "false");
  analysisRequests = 0;
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable");
  if (process.env.THROTTLE) await cdp.send("Emulation.setCPUThrottlingRate", { rate: Number(process.env.THROTTLE) });
  const before = await cdp.send("Performance.getMetrics");
  const res = await page.evaluate(async () => {
    const el = document.querySelector('[data-testid="scrubber"]');
    const max = Number(el.max);
    const work = [];
    const frames = [];
    const raf = () => new Promise((r) => requestAnimationFrame(() => r(performance.now())));
    await raf();
    for (let i = 0; i < 60; i++) {
      const v = (i * max) / 60;
      const t0 = performance.now();
      el.value = String(v);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      const t1 = performance.now();
      const t2 = await raf();
      work.push(t1 - t0);
      frames.push(t2 - t0);
    }
    const s = (a) => a.slice().sort((x, y) => x - y);
    const q = (a, p) => s(a)[Math.floor(p * (a.length - 1))];
    return {
      syncWork_ms: { median: q(work, 0.5), p90: q(work, 0.9), max: Math.max(...work) },
      toNextFrame_ms: { median: q(frames, 0.5), p90: q(frames, 0.9), max: Math.max(...frames) },
    };
  });
  await page.waitForTimeout(600);
  const after = await cdp.send("Performance.getMetrics");
  const m = (x, k) => x.metrics.find((e) => e.name === k)?.value ?? 0;
  res.scriptDuration_s = m(after, "ScriptDuration") - m(before, "ScriptDuration");
  res.layoutCount = m(after, "LayoutCount") - m(before, "LayoutCount");
  res.recalcStyleCount = m(after, "RecalcStyleCount") - m(before, "RecalcStyleCount");
  res.analysisRequestsAfterScrub = analysisRequests;
  console.log(JSON.stringify(res));
  await b.close();
})();
