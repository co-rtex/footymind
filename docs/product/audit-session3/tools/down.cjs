const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
(async () => {
  const b = await chromium.launch();
  const OUT = process.argv[2];
  const BASE = process.argv[3] ?? "http://127.0.0.1:5184";
  for (const [vp, size] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]]) {
    const p = await b.newPage({ viewport: size, deviceScaleFactor: 1 });
    await p.goto(BASE + "/");
    await p.waitForTimeout(2500);
    await p.screenshot({ path: `${OUT}/${vp}-14-api-down-landing.png` });
    await p.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
    await p.waitForTimeout(2500);
    await p.screenshot({ path: `${OUT}/${vp}-14b-api-down-lab.png` });
    await p.close();
  }
  await b.close();
})();
