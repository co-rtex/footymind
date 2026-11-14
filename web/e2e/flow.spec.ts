import { expect, test, type Page } from "@playwright/test";
import { dragPlayer, fetchAnalysis, openGoalPossession, pickDefender, waitForAnalysis } from "./helpers";

/** Review screenshots of the defender-response picker, only when SHOT_POLICY_DIR is set. */
async function shotPolicy(page: Page, name: string): Promise<void> {
  const dir = process.env.SHOT_POLICY_DIR;
  if (dir) await page.getByTestId("policy-picker").screenshot({ path: `${dir}/policy-picker-${name}.png` });
}

/**
 * Flagship flow against the real backend and real Metrica tracking data:
 * landing -> possession -> play -> pause -> analysis overlay -> drag a
 * defender -> scenario deltas -> difference view -> reset.
 */
test("replay, analyse and edit a scenario on real tracking data", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await openGoalPossession(page);
  await expect(page.getByTestId("source-attribution")).toContainText("Metrica Sports");

  // The lab opens paused on the first frame of the possession with analysis.
  const firstFrame = await waitForAnalysis(page);
  const clock = page.getByTestId("clock");
  const clockBefore = await clock.textContent();

  // Play: the canvas must change and the clock must advance.
  const canvas = page.getByTestId("pitch-canvas");
  const shotA = await canvas.screenshot();
  await page.getByTestId("play-toggle").click();
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Pause");
  await page.waitForTimeout(1600);
  const shotB = await canvas.screenshot();
  expect(Buffer.compare(shotA, shotB)).not.toBe(0);
  await expect(clock).not.toHaveText(clockBefore ?? "");

  // Pause with the keyboard shortcut, then step one frame with the arrow key.
  await page.locator("body").press("Space");
  await expect(page.getByTestId("play-toggle")).toHaveAttribute("aria-label", "Play");
  await page.locator("body").press("ArrowRight");

  // Analysis overlay for the paused frame.
  const frame = await waitForAnalysis(page);
  expect(frame).toBeGreaterThan(firstFrame);
  await expect(page.getByTestId("pitch-status")).toHaveText(/model overlay/);
  await expect(page.getByText("Estimated pitch control (model, uncalibrated)")).toBeVisible();
  await expect(page.getByTestId("model-key")).toContainText(/Option model \(calibrated on [\d,]+ real passes\)/);
  await expect(page.getByTestId("model-key")).toContainText("Physics heuristic (uncalibrated)");
  const optionRows = page.getByTestId("pass-options").locator("tbody tr");
  expect(await optionRows.count()).toBeGreaterThan(0);

  // Edit scenario: drag the defender nearest the ball 3 m towards the ball.
  const matchId = decodeURIComponent(page.url().split("/lab/")[1] ?? "").split(":")[0] ?? "";
  const a = await fetchAnalysis(page, matchId, frame);
  expect(a.attacking_team).not.toBeNull();
  const d = pickDefender(a);
  await page.getByRole("radio", { name: "Edit scenario" }).click();
  await expect(page.getByTestId("scenario-panel")).toBeVisible();
  const bx = a.ball_x ?? 0;
  const by = a.ball_y ?? 0;
  const len = Math.hypot(bx - d.x!, by - d.y!) || 1;
  const step = Math.min(3, len * 0.6);
  const to = { x: d.x! + ((bx - d.x!) / len) * step, y: d.y! + ((by - d.y!) / len) * step };
  await dragPlayer(page, { x: d.x!, y: d.y! }, to);

  // Scenario response: deltas, interval bars, agreement, evidence.
  const panel = page.getByTestId("scenario-panel");
  await expect(panel).toHaveAttribute("data-status", "success");
  const deltas = page.getByTestId("option-deltas");
  await expect(deltas).toBeVisible();
  const rows = deltas.locator("tbody tr");
  expect(await rows.count()).toBeGreaterThan(0);
  await expect(rows.first().locator("td").nth(2)).toHaveText(/^[+−]?\d\.\d\d$|^0\.00$/);
  await expect(deltas.locator("svg.ibars").first()).toBeVisible();
  await expect(page.getByTestId("scenario-headline")).toContainText("parameter settings agree");
  await expect(page.getByTestId("scenario-verdict")).toBeVisible();
  const evidence = page.getByTestId("evidence");
  await expect(evidence).toContainText("Evidence & assumptions");
  await expect(evidence).toContainText("not what would have happened in the match");
  await expect(panel.getByText(`#${d.jersey} ${d.team === "home" ? "Home" : "Away"} to (`)).toBeVisible();

  // Difference view, then response policy toggle re-runs the scenario.
  await page.getByRole("radio", { name: "Difference" }).click();
  await expect(page.getByText(/hypothetical − observed\), model, uncalibrated/)).toBeVisible();
  await expect(page.getByTestId("delta-model")).toContainText("Option model (calibrated on");
  await page.getByRole("radio", { name: "Physics heuristic" }).click();
  await expect(panel).toHaveAttribute("data-status", "success");
  await expect(page.getByTestId("delta-model")).toContainText("Physics heuristic (uncalibrated)");
  await page.getByTestId("policy-select").selectOption("marker_follows");
  await expect(panel).toHaveAttribute("data-status", "success");

  // Reset returns to the observed state.
  await page.getByTestId("reset-scenario").click();
  await expect(page.getByTestId("option-deltas")).toHaveCount(0);
  await expect(panel).toContainText("Drag any player");

  expect(errors).toEqual([]);
});

test("possession list loads with attribution and filters", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("source-attribution")).toContainText("Metrica Sports");
  const rows = page.getByTestId("possession-table").locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  const all = await rows.count();
  await page.getByLabel("Ends in a shot").check();
  await expect.poll(() => rows.count()).toBeLessThan(all);
  const shots = await rows.count();
  expect(shots).toBeGreaterThan(0);
  await expect(rows.locator(".badge--shot, .badge--goal")).toHaveCount(shots);
});

test("featured deep link opens a paused, physically supported scenario", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("featured-link").click();
  await expect(page).toHaveURL(/#\/at\/metrica-sample-1\?f=53708&move=home_9,50.7,-8.75/);
  const deltas = page.getByTestId("option-deltas");
  await expect(deltas).toBeVisible({ timeout: 30_000 });
  // The pass actually played (#17 -> #18) is the largest change and it is robust.
  await expect(deltas.locator("tbody tr").first()).toContainText("#18");
  await expect(deltas.locator("tbody tr").first()).toContainText("robust");
  await expect(page.getByText("Physically plausible edit")).toBeVisible();
});

test("defensive shadow layer ranks the suppressive defender at the featured frame", async ({ page }) => {
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("pitch-canvas")).toBeVisible();
  const replayAnalyse = page.getByRole("radio", { name: "Replay & analyse" });
  if ((await replayAnalyse.getAttribute("aria-checked")) !== "true") await replayAnalyse.click();
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");

  // Off by default, and keyboard-operable.
  const toggle = page.getByRole("checkbox", { name: "Defensive shadow (experimental)" });
  await expect(toggle).not.toBeChecked();
  await toggle.focus();
  await page.keyboard.press("Space");
  await expect(toggle).toBeChecked();

  const panel = page.getByTestId("shadow-panel");
  await expect(panel).toHaveAttribute("data-frame", "53708", { timeout: 30_000 });
  await expect(page.getByTestId("shadow-caveat")).toContainText("Not a defender rating.");
  // the served, provider-specific status (Metrica: claim C17), not the two-provider summary
  await expect(page.getByTestId("shadow-caveat")).toContainText("Validation for this match's provider");
  await expect(page.getByTestId("shadow-caveat")).toContainText("(C17)");
  const rows = page.getByTestId("shadow-row");
  const first = rows.first();
  await expect(first).toContainText("#9 Home");
  await expect(first).not.toContainText("within noise");
  await expect(first).toContainText(/\+0\.2\d/);
  expect(await rows.count()).toBeGreaterThan(1);
  await expect(panel.getByText("within noise").first()).toBeVisible();
  await expect(panel.getByText(/Assumptions \(\d+\)/)).toBeVisible();

  // Edit mode disables the layer, so it is never fetched there.
  await page.getByRole("radio", { name: "Edit scenario" }).click();
  await expect(toggle).toBeDisabled();
});

test("broadcast tracking: extrapolated SkillCorner positions are marked as estimates", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));

  // The landing groups providers and explains broadcast extrapolation, crediting SkillCorner.
  await page.goto("/");
  await expect(page.getByTestId("provider-note-skillcorner")).toContainText("extrapolated");
  await expect(page.getByTestId("provider-note-skillcorner").getByRole("link")).toHaveAttribute(
    "href",
    /github\.com\/SkillCorner\/opendata/,
  );

  const poss = (await (await page.request.get("/api/matches/skillcorner-2017461/possessions?min_passes=3")).json()) as {
    possession_id: string;
  }[];
  const pid = poss[0]?.possession_id;
  expect(pid).toBeTruthy();
  const replay = (await (await page.request.get(`/api/possessions/${encodeURIComponent(pid!)}/replay`)).json()) as {
    players_detected: boolean[] | null;
  };
  expect(replay.players_detected?.some((v) => v === false)).toBe(true);

  await page.goto(`/#/lab/${encodeURIComponent(pid!)}`);
  await expect(page.getByTestId("legend-estimated")).toContainText("Estimated (provider extrapolation, off camera)");
  await expect(page.getByTestId("pitch-canvas")).toHaveAttribute("aria-label", /provider estimates/);
  await expect(page.getByTestId("source-attribution")).toContainText("SkillCorner");

  // Replay renders and plays without errors, then analysis flags estimated positions.
  const canvas = page.getByTestId("pitch-canvas");
  const before = await canvas.screenshot();
  await page.getByTestId("play-toggle").click();
  await page.waitForTimeout(1200);
  await page.getByTestId("play-toggle").click();
  expect(Buffer.compare(before, await canvas.screenshot())).not.toBe(0);
  await waitForAnalysis(page);
  await expect(page.getByTestId("estimated-note")).toContainText("provider extrapolations");
  expect(errors).toEqual([]);
});

test("a Metrica-only match list shows no broadcast featured card and requests nothing it lacks", async ({ page }) => {
  // The real /api/matches, with every non-Metrica match removed (as on the README quick start).
  await page.route("**/api/matches", async (route) => {
    const res = await route.fetch();
    const all = (await res.json()) as { match_id: string }[];
    await route.fulfill({ response: res, json: all.filter((m) => m.match_id.startsWith("metrica-")) });
  });
  const failed: string[] = [];
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400) failed.push(`${r.status()} ${r.url()}`);
  });
  const asked: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/api/")) asked.push(r.url());
  });
  await page.goto("/");
  const section = page.getByTestId("featured-section");
  await expect(section.getByTestId("featured-link")).toBeVisible();
  await expect(section.getByTestId("featured-response-link")).toBeVisible();
  await expect(page.getByTestId("featured-broadcast-link")).toHaveCount(0);
  await expect(page.getByTestId("featured-missing")).toContainText("not loaded on this server");
  await expect(section.getByTestId("featured-facts").first()).toBeVisible({ timeout: 30_000 });
  expect(asked.filter((u) => u.includes("skillcorner"))).toEqual([]);
  expect(failed).toEqual([]);
});

test("featured broadcast scenario opens with detected key players and a robust change", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("featured-broadcast-link").click();
  await expect(page).toHaveURL(/#\/at\/skillcorner-1899585\?f=39177&move=home_25,22.46,-5.77/);
  const deltas = page.getByTestId("option-deltas");
  await expect(deltas).toBeVisible({ timeout: 30_000 });
  await expect(deltas.locator("tbody tr").first()).toContainText("#14");
  await expect(deltas.locator("tbody tr").first()).toContainText("robust");
  await expect(page.getByText("Physically plausible edit")).toBeVisible();
});

test("keyboard edit with a defender response: selector, simulated moves and static comparison", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");
  const a = await fetchAnalysis(page, "metrica-sample-1", 53708);

  // An attacker (not the carrier) with a defender within 8 m, so a marking response has someone to move.
  const defenders = a.players.filter((p) => p.team !== a.attacking_team && p.x !== null && p.y !== null);
  const attacker = a.players
    .filter((p) => p.team === a.attacking_team && p.x !== null && p.y !== null && !p.is_gk && p.player_id !== a.carrier_id)
    .map((p) => ({ p, d: Math.min(...defenders.map((q) => Math.hypot(q.x! - p.x!, q.y! - p.y!))) }))
    .filter((s) => s.d < 8)
    .sort((s, t) => s.d - t.d)[0]?.p;
  expect(attacker).toBeTruthy();

  // Keyboard-only path: mode switch, Player menu, move form (no canvas interaction).
  const analyseRadio = page.getByRole("radio", { name: "Replay & analyse" });
  await analyseRadio.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("scenario-panel")).toBeVisible();
  await page.getByTestId("player-menu").selectOption(attacker!.player_id);
  await expect(page.getByTestId("move-x")).toBeFocused();
  await page.getByTestId("move-x").fill((attacker!.x! - 2.5).toFixed(1));
  await page.getByTestId("move-y").fill(attacker!.y!.toFixed(1));
  await page.keyboard.press("Enter");
  const panel = page.getByTestId("scenario-panel");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  await expect(page.getByTestId("option-deltas")).toBeVisible();
  await expect(page.getByTestId("policy-static-hint")).toBeVisible();

  // The selector lists every policy the backend serves, static first, each option carrying the
  // verdict badge its served status implies (the static baseline has none).
  const served = (await (await page.request.get("/api/response-policies?match_id=metrica-sample-1")).json()) as {
    name: string;
    status: string;
  }[];
  const select = page.getByTestId("policy-select");
  const options = await select
    .locator("option")
    .evaluateAll((os) => os.map((o) => ({ value: (o as HTMLOptionElement).value, text: o.textContent ?? "", badge: o.getAttribute("data-badge") })));
  expect(options[0]).toMatchObject({ value: "none", badge: "" });
  expect(options.map((o) => o.value).sort()).toEqual(served.map((p) => p.name).sort());
  const badges = Object.fromEntries(options.map((o) => [o.value, o.badge]));
  expect(badges).toEqual({
    none: "",
    marker_follows: "observational",
    lane_cover: "observational",
    learned_innovation: "observational",
    learned: "forecaster",
    press_carrier: "not supported",
    zonal_shift: "not supported",
  });
  for (const o of options) if (o.badge) expect(o.text).toContain(`(${o.badge})`);
  await expect(page.getByTestId("policy-badge")).toHaveCount(0);

  // A response policy: provenance-labelled simulated moves (or an explicit "moved nobody"), its
  // verdict badge with the backend's own status text, and the static comparison.
  await select.selectOption("marker_follows");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  const comparison = page.getByTestId("policy-comparison");
  await expect(comparison).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("policy-badge")).toHaveText("observational");
  await expect(page.getByTestId("policy-status")).toContainText(served.find((p) => p.name === "marker_follows")!.status);
  await expect(page.getByTestId("policy-description")).toContainText("not a causal effect");
  await expect(page.getByTestId("policy-description")).not.toContainText("Not validated against observed defender movement");
  await expect(page.getByTestId("policy-fit-range")).toHaveCount(0);
  await shotPolicy(page, "marker-follows");
  await expect(page.getByTestId("policy-verdict")).not.toBeEmpty();
  const sims = page.getByTestId("simulated-move");
  if ((await sims.count()) > 0) {
    await expect(sims.first()).toContainText("simulated");
    await expect(sims.first()).toContainText("Marker follows response");
    await expect(comparison.getByRole("table")).toBeVisible();
  } else {
    await expect(page.getByTestId("no-simulated-moves")).toBeVisible();
  }
  await expect(page.getByTestId("evidence")).toContainText("not what would have happened in the match");

  // A rule the observational check does not support, then the learned reaction with its fitted range.
  await select.selectOption("zonal_shift");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  await expect(page.getByTestId("policy-badge")).toHaveText("not supported");
  await shotPolicy(page, "zonal-shift");
  await select.selectOption("learned_innovation");
  await expect(panel).toHaveAttribute("data-status", "success", { timeout: 30_000 });
  await expect(page.getByTestId("policy-badge")).toHaveText("observational");
  // the provider's own fitted range (served as fitted_range): 2.3 m on Metrica, 2.5 m on SkillCorner
  await expect(page.getByTestId("policy-fit-range")).toContainText(/under about 2\.[35] m in 1 s/);
  await expect(page.getByTestId("policy-fit-range")).toContainText("extrapolation");
  await shotPolicy(page, "learned-innovation");

  // Back to static: the comparison goes away and the hint returns.
  await select.selectOption("none");
  await expect(page.getByTestId("policy-static-hint")).toBeVisible();
  await expect(comparison).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("broken links explain themselves and lead back to the possession list", async ({ page }) => {
  await page.goto("/#/lab/metrica-sample-1%3Ap9999");
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("We could not find this possession");
  await expect(alert.getByRole("button", { name: "Retry" })).toHaveCount(0);
  await alert.getByRole("button", { name: "Browse possessions" }).click();
  await expect(page.getByTestId("possession-table")).toBeVisible();
});

test("why this moment: grouped statements with provenance, traceable to players, stable across frames", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  const why = page.getByTestId("why-panel");
  await expect(why).toHaveAttribute("data-frame", "53708");
  const a = (await (await page.request.get("/api/matches/metrica-sample-1/frames/53708/analysis?grid=false")).json()) as {
    explanations: { kind: string; text: string; provenance: string; players: string[] }[];
  };
  expect(a.explanations.length).toBeGreaterThan(0);

  // Every statement is shown, with its provenance label; caveats are always visible.
  const items = why.getByTestId("why-statement");
  await expect(items).toHaveCount(a.explanations.length);
  for (const s of a.explanations) {
    const item = items.filter({ hasText: s.text.replace(/\b(home|away)_(\d+)\b/g, (_m, t: string, n: string) => `#${n} ${t === "home" ? "Home" : "Away"}`) });
    await expect(item).toHaveCount(1);
    await expect(item).toContainText(s.provenance);
  }
  const caveats = a.explanations.filter((s) => s.kind === "caveat");
  if (caveats.length > 0) await expect(page.getByTestId("why-group-caveat").getByTestId("why-statement").first()).toBeVisible();

  // Focusing a statement that names players rings them on the pitch.
  const canvas = page.getByTestId("pitch-canvas");
  await page.mouse.move(0, 0);
  const before = await canvas.screenshot();
  const traceable = why.locator("button.why-text--trace").first();
  await traceable.focus();
  await expect.poll(async () => Buffer.compare(before, await canvas.screenshot())).not.toBe(0);
  await traceable.blur();

  // Stepping frames updates the panel without shrinking it (no layout jump).
  const h0 = (await why.boundingBox())!.height;
  await page.locator("body").press("ArrowRight");
  await expect(why).toHaveAttribute("data-frame", "53709");
  expect((await why.boundingBox())!.height).toBeGreaterThanOrEqual(h0 - 0.5);
  expect(errors).toEqual([]);
});

test("similar moments: on demand, observed outcomes with links, quiet empty states", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let similarCalls = 0;
  page.on("request", (r) => {
    if (/\/api\/matches\/[^/]+\/frames\/\d+\/similar/.test(r.url())) similarCalls++;
  });
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");
  const panel = page.getByTestId("similar-panel");
  // Not fetched on load or while scrubbing.
  await page.locator("body").press("ArrowRight");
  await page.locator("body").press("ArrowLeft");
  await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "53708");
  expect(similarCalls).toBe(0);

  await page.getByTestId("similar-find").click();
  await expect(panel).not.toHaveAttribute("data-status", "loading", { timeout: 30_000 });
  const status = await panel.getAttribute("data-status");
  if (status === "empty") {
    // No index built for this provider on this server: a quiet note, never an error.
    await expect(page.getByTestId("similar-empty-no-index")).toBeVisible();
  } else {
    expect(status).toBe("success");
    const res = (await (await page.request.get("/api/matches/metrica-sample-1/frames/53708/similar?k=8")).json()) as {
      neighbours: { completed: boolean }[];
      validation: { knn_beats_base_rate: boolean | null } | null;
    };
    const n = res.neighbours.length;
    const m = res.neighbours.filter((x) => x.completed).length;
    await expect(page.getByTestId("similar-headline")).toContainText(`In ${n} similar`);
    await expect(page.getByTestId("similar-headline")).toContainText(`${m} ${m === 1 ? "pass was" : "passes were"} completed`);
    await expect(page.getByTestId("similar-disclaimer")).toContainText("not a prediction");
    const validated = res.validation?.knn_beats_base_rate === true;
    await expect(page.getByTestId("similar-unvalidated")).toHaveCount(validated ? 0 : 1);
    await expect(page.getByTestId("similar-validated-pill")).toHaveCount(validated ? 1 : 0);
    if (validated) await expect(page.getByTestId("similar-validated")).toContainText("Validated on held-out matches");
    const items = page.getByTestId("similar-item");
    await expect(items).toHaveCount(n);
    await expect(items.first()).toContainText("observed");
    // Each row carries the neighbour's match clock; opening it lands on the same clock in the lab.
    const context = items.first().getByTestId("similar-context");
    await expect(context).toHaveText(/^P\d \d\d:\d\d/);
    const rowClock = /^P\d (\d\d:\d\d)/.exec((await context.textContent()) ?? "")?.[1];
    const link = items.first().getByRole("link", { name: /^Open / });
    await expect(link).toHaveAttribute("href", /^#\/at\/metrica-sample-[23]\?f=\d+$/);
    const href = (await link.getAttribute("href"))!;
    await link.click();
    await expect(page).toHaveURL(new RegExp(href.replace(/[?]/g, "\\?") + "$"));
    await expect(page.getByTestId("pitch-canvas")).toBeVisible();
    await expect(page.getByTestId("clock")).toHaveText(rowClock!);
    await expect(page.getByTestId("source-attribution")).toContainText("Metrica Sports");

    // A frame with no pass decision within 2 s: a quiet empty state, not an error.
    await page.goto("/#/lab/metrica-sample-1%3Ap0167?f=54500");
    await expect(page.getByTestId("analysis-panel")).toHaveAttribute("data-frame", "54500");
    await page.getByTestId("similar-find").click();
    await expect(page.getByTestId("similar-empty-no-decision")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("similar-panel").getByRole("alert")).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test("team names: SkillCorner matches are named, Metrica stays anonymised", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const sc = (await (await page.request.get("/api/matches/skillcorner-1874553")).json()) as {
    home_label: string;
    away_label: string;
    competition: string | null;
    date: string | null;
  };
  const named = sc.home_label !== "Home" || sc.away_label !== "Away";
  const title = [named ? `${sc.home_label} vs ${sc.away_label}` : null, sc.competition, sc.date].filter(Boolean).join(" · ");

  await page.goto("/");
  const card = page.getByTestId("match-card-skillcorner-1874553");
  if (named) {
    await expect(card).toContainText(`${sc.home_label} vs ${sc.away_label}`);
    if (sc.competition) await expect(card).toContainText(sc.competition);
  }
  await card.click();
  const rows = page.getByTestId("possession-table").locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  if (named) {
    await expect(page.getByTestId("possessions-match")).toHaveText(title);
    await expect(page.getByRole("radio", { name: sc.home_label, exact: true })).toBeVisible();
    await expect(rows.first().locator(".team-chip")).toHaveText(new RegExp(`^(${sc.home_label}|${sc.away_label})$`));
  }

  await rows.first().getByRole("link").click();
  await expect(page.getByTestId("pitch-canvas")).toBeVisible();
  if (named) {
    await expect(page.getByTestId("match-title")).toHaveText(title);
    await expect(page.locator(".h-page .team-chip")).toHaveText(new RegExp(`^(${sc.home_label}|${sc.away_label})$`));
    await expect(page.getByRole("group", { name: "Legend" })).toContainText(sc.home_label);
    await expect(page.getByRole("group", { name: "Legend" })).toContainText(sc.away_label);
    await waitForAnalysis(page);
    await expect(page.getByTestId("analysis-panel")).toContainText(new RegExp(`#\\d+ (${sc.home_label}|${sc.away_label})`));
  }

  // Metrica is anonymised: Home/Away only, titled by its source name.
  await page.goto("/#/lab/metrica-sample-1%3Ap0163?f=53708");
  await expect(page.getByTestId("match-title")).toHaveText("Metrica Sports Sample Game 1");
  await expect(page.locator(".h-page .team-chip")).toHaveText("Away");
  await expect(page.getByRole("group", { name: "Legend" })).toContainText("Home");
  expect(errors).toEqual([]);
});
