# Product audit, session 3 (2026-11-14)

The web lab was run against the real API and real processed data (Metrica and SkillCorner,
pipeline v2) and driven with Playwright. Every state below was screenshotted at device scale
factor 1 at three viewports: desktop 1440×900, tablet 834×1112 and phone 390×844. Each
screenshot was inspected by eye. This document records what a non-technical football viewer
would trip on, what was fixed and what is still open.

Evidence images are in `docs/product/audit-session3/{before,after}/`. They are JPEG copies,
and the phone crops are cut from full-page captures. The scripts that produced them are in
`docs/product/audit-session3/tools/`; see the end of this document for how to run them.

## Method

| Step | Tool | Notes |
|---|---|---|
| States | `tools/audit.cjs` | landing, match selection, lab paused + timeline, player card, pass-lane highlight, Defensive Shadow, edit mode, dragging a defender, scenario, difference view, defender response, evidence panel, featured deep link, bad possession id, bad route, `#/at` frame with no possession |
| API stopped | `tools/down.cjs` | a second Vite server proxies `/api` to a dead port, which is what a user sees when the backend is down in development |
| Keyboard | `tools/kbd.cjs` | records the Tab order in edit mode |
| Accessibility | `tools/axe.cjs` | axe-core 4.10.3 on the landing, the lab with the Shadow layer, and a scenario |
| Scrubbing cost | `tools/perf.cjs` | 60 scrubber `input` events: synchronous work per event, time to the next painted frame, and CDP `ScriptDuration`. Run at 1× and at 4× CPU throttling, on dev and production builds |
| Response policies | `tools/policy.cjs`, `tools/future.cjs` | the real backend (`none` and `marker_follows`), and the future contract mocked with **synthetic** route data (see Goal 2) |

## Findings

Severity: **high** means the user cannot finish a task or is misled. **Medium** means the
user finishes the task with avoidable confusion or extra work. **Low** means friction or
polish.

| # | Problem | Severity | Evidence | Status |
|---|---|---|---|---|
| F1 | **Keyboard users could not edit a scenario.** The only way to choose a player was to click the canvas. The panel told keyboard users to "select a player and use the move form", but no control did that: in edit mode, Tab cycled through the back button, the mode switch, four layer checkboxes, the view switch, the policy and the value model, then left the page. | high (accessibility) | Tab order recorded by `tools/kbd.cjs` (before: no player control; after: item 7 is the Player menu) | **Fixed.** A "Player" menu above the pitch, grouped by team, lists the players present at the frame. In edit mode, choosing a player focuses the x field of the move form. A new e2e test runs the whole edit with the keyboard. |
| F2 | **The move form rejected most coordinates.** The inputs used `step="0.5"`, so the browser refused any value that is not a multiple of 0.5, including the prefilled observed position (for example 46.7): "Please enter a valid value". The keyboard path to a scenario was broken even after F1. | high | Found by the new e2e test (Playwright failure screenshot) | **Fixed** (`step="0.1"`). |
| F3 | **On desktop, the pitch scrolled away while reading results.** The scenario and analysis column is about twice the viewport height. Hovering a row in the pass-options or delta table highlights its lane on a pitch that is no longer on screen. | high | `before/desktop-11-evidence.jpg` | **Fixed.** At 1181 px and wider, the analysis column is sticky and scrolls on its own, so the pitch stays in view. See `after/desktop-11-evidence.jpg` and `after/desktop-16c-policy-comparison.jpg`. |
| F4 | **On tablet and phone, a drag seemed to do nothing.** The result appears roughly 1,000 px below the pitch, and the only visible change is the small "Hypothetical frame" chip. | high | `before/tablet-09-scenario.jpg` | **Fixed.** In stacked layouts, a one-line result under the transport ("Largest change: pass to #16 Away −0.12 · Full comparison ↓") scrolls to the panel and focuses its heading. See `after/tablet-09-scenario.jpg`. |
| F5 | **A stale result could show under a new choice.** While a re-run was in flight, the panel showed the previous result as current. For the new policy UI, this would have described the old policy's moves under the new policy's name. The first Goal 2 build did exactly that on tablet and phone ("moved no defender" while the request was loading). | high (correctness) | `tools/policy.cjs` output during development | **Fixed.** Each result is keyed by its request. Statements that depend on the policy (simulated moves, the comparison) appear only for the result of the current request. |
| F6 | **Error messages were technical or useless.** With the API stopped, the user saw "HTTP 500: Internal Server Error" (the dev proxy's empty 500). A bad or stale possession link showed "HTTP 404: unknown possession …" with a Retry button that cannot help. | medium | `before/desktop-14-api-down-landing.jpg`, `before/phone-14b-api-down-lab.jpg`, `before/desktop-13-bad-possession.jpg` | **Fixed.** Errors are written in plain language ("The FootyMind server is not responding…", "We could not find this possession. The link may be mistyped, or it may come from an older version of the processed data.") and keep the technical line in small monospace. Retry is shown only when retrying can help. Lab errors offer "Browse possessions". On a phone, the block wraps. See the matching `after/` images. |
| F7 | **The phone delta table clipped the Agreement column** ("100% r…") with no sign that it scrolls. | medium | `before/phone-deltas-clipped.jpg` | **Fixed.** Below 480 px the interval-bar column is hidden; the observed and hypothetical values stay in their own columns. See `after/phone-deltas.jpg`. |
| F8 | **On a phone, choosing a match changed nothing visible.** Twenty SkillCorner cards were stacked in one column (about 1,700 px) above the possession list. | medium | `before/phone-landing-matches.jpg` | **Fixed.** On phones, the broadcast matches use two compact columns, and choosing a match brings the possession list's heading into view (instantly, with no animation). The landing page is about 520 px shorter. See `after/phone-landing-matches.jpg`. |
| F9 | **The match names mean nothing to a viewer** ("Match 1874553"). There are no team names or dates, and `home_label`/`away_label` are "Home"/"Away". | medium | `before/desktop-02-match-selected.jpg` | **Open: needs the backend** (see "Backend changes needed"). |
| F10 | **At 1440×900, the legend is below the fold in the lab.** Without scrolling, the brown and teal pitch-control colours are unexplained. | medium | `before/desktop-03-lab-paused.jpg` | Open. The proposed fix is a compact colour key in the stage toolbar. It was deferred to avoid crowding the toolbar, which now holds the Player menu. |
| F11 | **The player card shows the raw id (`away_17`) and centre-spot coordinates.** It gives no football description of where the player is. | low | `before/desktop-04-player-inspector.jpg` | Open. The proposal is a zone line ("in the opponent's box, left half-space"); this needs a product decision on the zone grid. |
| F12 | **Pass-option headers (Phys., Lane, Ctrl, Fwd) are explained only by hover tooltips**, which touch screens never show. | low | `before/desktop-03-lab-paused.jpg` | Open. The caption carries the full text for screen readers. A visible "How to read this table" disclosure is proposed. |
| F13 | **axe flagged contrast in the locked transport.** The period and frame text in edit mode measured 3.55:1 and 2.58:1, because the whole transport was dimmed with opacity. | low (accessibility) | `tools/axe.cjs` output (before) | **Fixed.** Only the controls are dimmed, and the clock stays readable. axe now reports **no violations** on the landing, lab + Shadow, and scenario states. |
| F14 | **Scrubbing cost.** At 1× CPU, each scrub step paints in the next frame (median 16.6 ms), so scrubbing feels fine. At 4× CPU throttling (a mid-range phone), each step re-rendered the whole lab, including the analysis panel, events list and legend. | low (performance) | numbers below | **Partly fixed.** The side panels are memoised and their callbacks are stable. The remaining cost is the lab view itself re-rendering on every frame index. |
| F15 | **The opponent-response control (two segmented buttons) would not fit six policies**, and it did not say which policies are validated. | medium (Goal 2) | `before/desktop-09-scenario.jpg` | **Replaced** by the "Defender response" selector (Goal 2). |
| F16 | **The Shadow caveat claimed partial validation without mentioning that it does not replicate on SkillCorner.** The same caveat appears on broadcast matches. | medium (honesty) | `before/desktop-06-shadow.jpg` (caveat text) | **Fixed.** The caveat now says that on SkillCorner broadcast data the Shadow does no better than chance (CLAIMS_LEDGER C29). The Metrica statement stays (C17). |
| F17 | **The phone pitch is about 330 px wide**, and player dots are hard to tap or drag. | low | `before/phone-03-lab-paused.jpg` | **Mitigated.** The Player menu and move form are an alternative to dragging. The pitch itself is unchanged. |
| F18 | **Simulated-move reasons used raw ids** ("marker of away_18 follows…"). | low | development screenshot | **Fixed.** Known ids are rewritten as "#18 Away". |

No horizontal page overflow was found at any viewport in any state (`scrollWidth ≤ clientWidth`
for every screenshot). No page errors were logged. The only console errors are the expected
404s from the deliberately broken links and from `/api/response-policies` on the current
backend. The latter is requested once per match per session.

### Scrubbing measurements (F14)

Each run sends 60 scrubber `input` events across the replay at the featured frame
(`metrica-sample-1:p0163`). "Script" is the CDP `ScriptDuration` over the 60 steps.

| Build | CPU | Before: median ms to next frame / script | After: median ms to next frame / script |
|---|---|---|---|
| dev (React development build) | 1× | 16.6 / 0.43 s | 16.6 / 0.35 s |
| dev | 4× | 44 / 1.63 s | 40 / 1.24 s |
| production (`vite build` + `vite preview`) | 4× | 35 / 0.85 s | 27 / 0.55 s |

Next step, if needed: during a scrub drag, update the React frame index once per animation
frame rather than once per input event. The canvas is already drawn imperatively.

## Goal 2: defender response policies

The UI follows the contract agreed for the parallel backend work, and it tolerates fields that
are absent on the current backend.

* **Selector.** "Defender response" is a native `<select>` filled from
  `GET /api/response-policies?match_id=…`. When the endpoint is missing (404 on the current
  backend), it falls back to `none` and `marker_follows`. The 404 is remembered per match. The
  static baseline is listed first as "Static (no response)". Unknown policy names are shown in
  readable form rather than hidden.
* **Experimental badge.** Any policy other than static with `validated: false` gets an
  "experimental" badge and the sentence "Not validated against observed defender movement".
  A validated policy shows its `validation_ref` instead. The response's own `policy` object
  takes precedence over the list entry, and its `assumptions` appear under Evidence &
  assumptions.
* **Simulated moves.** On the pitch, a simulated move is a white (the simulated provenance
  colour) short-dashed arrow with a "sim" tag. A user move is a violet dashed arrow. In the
  panel, each simulated move is a row with a `SIMULATED` provenance badge, the policy name and
  the backend's reason. When a policy moves nobody, the panel says so.
* **Static vs policy comparison.** When a non-static policy is chosen, the lab also runs the
  same edit with `response_policy: "none"`; the result is usually already cached. The
  comparison shows a plain-language verdict ("The conclusion depends on how defenders respond:
  1 of 10 pass options change direction…", "No pass option changes direction, but the size of
  the changes differs by up to 0.05", or "Little difference…") and a table of the five largest
  changes (Static Δ, Response Δ, Gap, and a "direction flips" marker), which expands to all
  options. It is labelled as a model estimate. The counterfactual disclaimer stays in the
  evidence section.
* **Contract checks.** `parse.ts` checks `policy` when it is present (name, description,
  assumptions as strings, validated as a boolean, `validation_ref` as a string, null or
  absent), the policy list (no duplicate names), and each movement's numeric coordinates.
* **Tests.** Vitest covers the parser, the client (the policy URL, the 404 fallback, and the
  unreachable proxy), the comparison logic, the plain-language verdicts, the error
  descriptions and the panel (fallback list, experimental badge, validated reference, unknown
  policy, simulated rows, comparison). The new e2e test checks the selector, the static-first
  ordering, `marker_follows`, the simulated moves (or the explicit "moved no defender"), the
  experimental badge (unless the backend reports the policy as validated) and the comparison.
  It passes on the current backend and is written to pass on the extended one.
* **Future contract, mocked.** `tools/future.cjs` intercepts the two endpoints with **synthetic**
  data: a made-up policy list, and a made-up `policy` object added to real `marker_follows`
  responses. It confirmed that the selector lists all policies, that the badge is on for
  unvalidated policies and off for a validated one, and that the policy assumptions render.
  These images (`after/desktop-17-future-contract-*.jpg`) show mocked values, not results.

## Goal 3: Vitest upgrade, decided to upgrade

* `npm audit` before the upgrade reported 10 findings (2 critical, 8 moderate), all in dev
  dependencies:
  * critical: `tinypool` ≤ 2.1.1 (prototype pollution leading to RCE), reached through Vitest;
  * moderate: `@vitest/mocker` (path traversal and mock-registration code injection);
  * moderate: `minimatch` under `typescript-eslint`.
* `npm view vitest` gives `latest` 5.1.0 (released 2026-11-02). It requires Vite
  `^6.4 || ^7 || ^8` (installed: 6.4.6), Node `^22.12 || ^24 || >=26` (local 22.23.4; CI and
  Docker use Node 22) and `@types/node ^22 || >=24`.
* The upstream migration guide could not be read from this session, because `vitest.dev` is
  outside the session's network allowlist and the GitHub repository is not attached. The
  decision rests on the full suite instead: after `npm install -D vitest@5.1.0`, `typecheck`,
  `lint`, **104/104 vitest**, `build` and **8/8 Playwright e2e** pass. No configuration or test
  changes were needed. The project uses no snapshot, coverage, browser-mode or workspace
  features, which are the areas majors usually break.
* After the upgrade, `npm audit` reports **7 moderate** findings, all in the `minimatch`
  chain under `typescript-eslint` 8.72.2, which is the latest release. The fix it suggests
  (8.36.0) is a downgrade, so those findings are left for an upstream release. They are
  dev-only and affect linting only.

## Backend changes needed (not made here; `src/` is owned by others)

1. **Team names in `MatchSummary`** (F9). Populate `home_label` and `away_label` from the
   SkillCorner match metadata, and add a date or round if one is available. Metrica is
   anonymised, so "Home"/"Away" is correct there.
2. **The response-policy contract** as agreed: `GET /api/response-policies?match_id=` and
   `ScenarioOut.policy`. The UI already handles both. Please keep `name` values stable,
   because they appear in deep-link state and tests.
3. **Optional: scenario `reason` text with player labels.** The UI rewrites `home_N`/`away_N`
   itself, so this is optional.
4. **Optional: `GET /api/response-policies` without a `match_id`**, for provider-independent
   policies. The UI calls it per match today.

## Reproducing the audit

```bash
export FOOTYMIND_DATA_DIR=/path/to/data
uv run footymind serve --port 8030 &
(cd web && FOOTYMIND_API=http://127.0.0.1:8030 npx vite --port 5183 --strictPort &)
# API-down states: a second dev server proxying to a dead port
(cd web && FOOTYMIND_API=http://127.0.0.1:8099 npx vite --port 5184 --strictPort &)
export WEB=$PWD/web
node docs/product/audit-session3/tools/audit.cjs /tmp/shots          # all states, 3 viewports
node docs/product/audit-session3/tools/down.cjs /tmp/shots           # API stopped
node docs/product/audit-session3/tools/policy.cjs /tmp/shots         # marker_follows with a simulated move
THROTTLE=4 node docs/product/audit-session3/tools/perf.cjs http://127.0.0.1:5183
# axe: install axe-core outside the repository and point the script at it
npm i --prefix /tmp/axe axe-core@4.10.3
AXE_JS=/tmp/axe/node_modules/axe-core/axe.min.js node docs/product/audit-session3/tools/axe.cjs
```
