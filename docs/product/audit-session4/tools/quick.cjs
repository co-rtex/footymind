// quick look: node quick.cjs <outDir> <vp> <hash> <name> [full] [action]
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const [OUT, vp, hash, name, full, action] = process.argv.slice(2);
const VP = { desktop: { width: 1440, height: 900 }, tablet: { width: 834, height: 1112 }, phone: { width: 390, height: 844 } };
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: VP[vp], deviceScaleFactor: 1 })).newPage();
  const msgs = [];
  page.on("pageerror", (e) => msgs.push(`pageerror: ${e.message}`));
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && msgs.push(`${m.type()}: ${m.text()}`));
  await page.goto((process.env.BASE ?? "http://127.0.0.1:5190") + "/" + hash);
  await page.waitForTimeout(3500);
  if (action) {
    for (const a of action.split(";")) {
      const i = a.indexOf("="); const kind = a.slice(0, i); const arg = a.slice(i + 1);
      if (kind === "click") await page.locator(arg).first().click();
      if (kind === "wait") await page.waitForTimeout(Number(arg));
      if (kind === "scroll") await page.locator(arg).first().scrollIntoViewIfNeeded();
      if (kind === "pick") {
        // pick=<select selector>|<option index>
        const [sel, n] = arg.split("|");
        const v = await page.locator(sel).first().locator("option").nth(Number(n)).getAttribute("value");
        await page.locator(sel).first().selectOption(v);
      }
      if (kind === "key") await page.keyboard.press(arg);
      if (kind === "y")await page.evaluate((y) => window.scrollTo(0, y), Number(arg));
    }
  }
  await page.mouse.move(1, 1);
  await page.screenshot({ path: path.join(OUT, `${name}.png`), fullPage: full === "1" });
  const ov = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  console.log(JSON.stringify({ name, overflowX: ov, msgs }));
  await browser.close();
})();
