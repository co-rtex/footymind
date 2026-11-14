// Screenshot: keyboard-made attacker move + Marker follows response (simulated defender move).
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5183";
(async () => {
  const b = await chromium.launch();
  for (const [vp, size] of [["desktop", { width: 1440, height: 900 }], ["tablet", { width: 834, height: 1112 }], ["phone", { width: 390, height: 844 }]]) {
    const page = await b.newPage({ viewport: size, deviceScaleFactor: 1 });
    await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
    await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "53708");
    const a = await (await page.request.get(`${BASE}/api/matches/metrica-sample-1/frames/53708/analysis?grid=false`)).json();
    const defs = a.players.filter((p) => p.team !== a.attacking_team && p.x !== null);
    const att = a.players
      .filter((p) => p.team === a.attacking_team && p.x !== null && !p.is_gk && p.player_id !== a.carrier_id)
      .map((p) => ({ p, d: Math.min(...defs.map((q) => Math.hypot(q.x - p.x, q.y - p.y))) }))
      .filter((s) => s.d < 8)
      .sort((s, t) => s.d - t.d)[0].p;
    await page.getByRole("radio", { name: "Edit scenario" }).click();
    await page.getByTestId("player-menu").selectOption(att.player_id);
    await page.getByTestId("move-x").fill((att.x - 2.5).toFixed(1));
    await page.getByTestId("move-y").fill(att.y.toFixed(1));
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
    await page.getByTestId("policy-select").selectOption("marker_follows");
    await page.getByTestId("policy-comparison").waitFor({ timeout: 30000 });
    await page.getByTestId("player-menu").selectOption("");
    await page.mouse.move(0, 0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${OUT}/${vp}-16-policy-simulated.png` });
    await page.getByTestId("policy-picker").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-16b-policy-panel.png` });
    await page.getByTestId("policy-comparison").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/${vp}-16c-policy-comparison.png` });
    const sims = await page.getByTestId("simulated-move").allTextContents();
    console.log(vp, JSON.stringify(sims), await page.getByTestId("policy-verdict").textContent());
    await page.close();
  }
  await b.close();
})();
