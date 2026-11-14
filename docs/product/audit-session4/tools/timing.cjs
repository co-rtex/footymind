const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  const t0 = Date.now();
  p.on("console", (m) => console.log("console", m.type(), m.text()));
  p.on("response", (r) => r.url().includes("/api/") && console.log(Date.now() - t0, r.status(), r.url().slice(22, 120)));
  p.on("requestfailed", (r) => console.log("FAILED", Date.now() - t0, r.url().slice(22, 120), r.failure()?.errorText));
  await p.goto("http://127.0.0.1:5190/" + (process.argv[2] ?? ""));
  await p
    .waitForSelector(process.argv[3] ?? '[data-testid="featured-facts"]', { timeout: 30000 })
    .then(() => console.log("found after", Date.now() - t0))
    .catch(() => console.log("timeout"));
  await b.close();
})();
