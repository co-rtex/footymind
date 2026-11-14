// Ad-hoc check of the UI against the FUTURE policy contract, with SYNTHETIC route mocks:
// GET /api/response-policies returns a made-up list, and POST /api/scenarios for a new policy is
// forwarded to the real API as marker_follows with a made-up `policy` object added. Not real results.
const path = require("path");
const { chromium } = require(path.join(process.env.WEB, "node_modules/playwright"));
const OUT = process.argv[2];
const BASE = process.argv[3] ?? "http://127.0.0.1:5183";
const LIST = [
  { name: "none", description: "Static baseline (synthetic mock description).", assumptions: [], validated: false, validation_ref: null },
  { name: "marker_follows", description: "Marker rule (synthetic mock description).", assumptions: ["mock assumption"], validated: false, validation_ref: null },
  { name: "press_carrier", description: "Nearest defender presses the carrier (synthetic mock).", assumptions: ["mock: press at 5 m/s"], validated: false, validation_ref: null },
  { name: "learned", description: "Learned response (synthetic mock).", assumptions: ["mock"], validated: true, validation_ref: "CLAIMS_LEDGER C99 (mock)" },
];
(async () => {
  const b = await chromium.launch();
  const page = await b.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/api/response-policies**", (r) => r.fulfill({ json: LIST }));
  await page.route("**/api/scenarios", async (route) => {
    const req = JSON.parse(route.request().postData());
    const asked = req.response_policy;
    const fwd = { ...req, response_policy: ["none", "marker_follows"].includes(asked) ? asked : "marker_follows" };
    const res = await route.fetch({ postData: JSON.stringify(fwd) });
    const body = await res.json();
    body.policy = LIST.find((p) => p.name === asked) ?? null;
    await route.fulfill({ response: res, json: body });
  });
  await page.goto(BASE + "/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await page.waitForFunction(() => document.querySelector('[data-testid="analysis-panel"]')?.getAttribute("data-frame") === "53708");
  await page.getByRole("radio", { name: "Edit scenario" }).click();
  await page.getByTestId("player-menu").selectOption("away_18");
  await page.getByTestId("move-x").fill("44.2");
  await page.getByTestId("move-y").fill("-5.3");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector('[data-testid="scenario-panel"]')?.getAttribute("data-status") === "success", null, { timeout: 30000 });
  const opts = await page.getByTestId("policy-select").locator("option").allTextContents();
  console.log("options:", JSON.stringify(opts));
  for (const name of ["press_carrier", "learned"]) {
    await page.getByTestId("policy-select").selectOption(name);
    await page.getByTestId("policy-comparison").waitFor({ timeout: 30000 });
    const badge = await page.getByTestId("policy-experimental").count();
    const desc = await page.getByTestId("policy-description").textContent();
    const pa = await page.getByTestId("policy-assumptions").textContent().catch(() => "(none)");
    console.log(name, "badge:", badge, "| desc:", desc, "| assumptions:", pa);
    await page.getByTestId("player-menu").selectOption("");
    await page.getByTestId("policy-picker").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUT}/desktop-17-future-contract-${name}.png` });
  }
  console.log("errors:", JSON.stringify(errors));
  await b.close();
})();
