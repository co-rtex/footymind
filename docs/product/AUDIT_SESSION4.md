# Product audit, session 4 (2026-11-14): the first-visit journey

Question: can a first-time visitor, without reading the paper, understand what "The Invisible
Game" means, open a real situation, watch it, see why it matters, inspect the evidence, explore
similar moments, move a defender, compare estimates (including the defender response) and know
what the model does and does not know?

The web lab was driven with Playwright against the real API and real processed data (Metrica and
SkillCorner, pipeline v2, similarity indexes built) at 1440×900, 834×1112 and 390×844, device
scale factor 1. Every screenshot was inspected by eye. The browser console was recorded in every
state. Earlier findings F1–F18 are in [`AUDIT_SESSION3.md`](AUDIT_SESSION3.md); F10–F12 were
still open and are closed here.

Evidence is in `docs/product/audit-session4/{before,after}/`, as JPEG copies of the DPR 1
captures. Full-page landings are downscaled by 50%. The scripts are in
`docs/product/audit-session4/tools/`; see the end of this document.

## The journey, before and after

| Step | Before | After |
|---|---|---|
| 1. Understand the idea | Hero "Quantify the invisible game." with one generic sentence. Two dense text cards followed, then 23 match cards and a 106-row table: the landing was 6,508 px tall at 1440 px (`before/desktop-01-arrive-full.jpg`). | "The Invisible Game", plus one contrast: event data records the pass, not the space, pressure or other options. Four tiles (Space, Pressure, Passing options, What if?) each carry their provenance label. The page is 3,484 px tall (6,508 before; tablet 4,816 from 6,807; phone 7,163 from 9,508) (`after/desktop-01-arrive-full.jpg`). |
| 2. Pick a featured situation | Two featured cards with hard-coded numbers (see G2). | Two featured cards. The story is curated text; the numbers are computed live by the API and labelled. There is a three-step guided tour card, and on stacked layouts a CTA in the hero. |
| 3. Watch the observed possession | The featured link opened straight into edit mode, difference view, so the observed play was never shown. | Tour step 1, *Watch*, plays the possession from the start of the window. On stacked layouts the pitch is scrolled into view. |
| 4. Why this moment | Present, but the tour-less path skipped it (edit mode hides it). | Tour step 2, *Understand*: pauses on the decision frame in analyse mode, with buttons to Why this moment, Pass options and Similar moments. |
| 5. Inspect the evidence | Pass-option columns were explained only in tooltips (F12). | A visible "How to read this table" disclosure under both tables. |
| 6. Similar moments | About 1,900 px down the scrolling analysis column, with no cue (`before/desktop-08-similar.jpg`). | Jump links at the top of the analysis column, and a tour button. |
| 7. Move a defender | Featured deep link only. | Tour step 3, *What if?*, applies the curated move (hypothetical) and opens the difference view. Dragging and the keyboard move form are unchanged. |
| 8. Compare, with defender response | Choosing any response policy said "moved no defender", with no reason (G4, `before/desktop-11-response.jpg`). | The panel and the tour say why: the response models react to a moved attacker or ball carrier, and this edit moves only a defender. |
| 9. What the model knows | Labels were consistent, but no single key explained them. The legend drew provenance in the home team's amber. The colour scale sat below the fold (F10). | A five-label provenance key on the landing and in the legend, with neutral glyphs; a compact colour key in the pitch margin; an estimated ball drawn hollow with an "est." tag. |

## Findings

Severity: **high** means the user cannot finish a task or is misled. **Medium** means the user
finishes with avoidable confusion or extra work. **Low** means friction or polish.

| # | Problem | Severity | Evidence | Status |
|---|---|---|---|---|
| G1 | **The landing did not explain "the invisible game"** or how to start. The hero restated the brand; the featured cards were paragraphs of metrics; a 106-row table made the page 6.5k px tall. | high | `before/desktop-01-arrive.jpg`, `before/desktop-01-arrive-full.jpg`, `before/phone-01-arrive.jpg` | **Fixed.** The new hierarchy runs idea → guided tour → featured situations → "How to read what you see" → explore. The possession list shows 20 rows plus "Show all N possessions", and an empty filter result offers "Clear filters". `after/desktop-01-arrive.jpg`, `after/tablet-01-arrive.jpg`, `after/phone-01-arrive.jpg` |
| G2 | **The featured cards quoted numbers typed into the source** (Shadow 0.27 / 0.21, "0.45 to 0.76", "0.60 to 0.80", 24 members, "9 of 22", P1 35:48). They had already drifted: the broadcast card said 0.60 → 0.80, while the API (pipeline v2) returns **0.59 → 0.81**. | high (data honesty) | `before/desktop-01-arrive.jpg` against `after/desktop-02-featured.jpg` | **Fixed.** `lib/featured.ts` holds only the curated identity (match, frame, pass, defender, move) and qualitative text. A test fails if a story contains a decimal or a percentage. Each card makes the lab's own scenario request (`lib/scenario.ts`) and reads the Defensive Shadow, the clock, agreement, displacement and observed/estimated counts from the responses. An e2e test checks that the landing's numbers equal the lab's delta row. Loading shows skeletons; a failure shows "Live numbers unavailable" with Retry, and the links stay usable. |
| G3 | **There was no guided path.** The featured link landed first-timers in a dense edit mode, and the observed play and the reasons were never shown. | high | `before/desktop-03-featured-opened.jpg`, `before/phone-03-featured-opened.jpg` | **Fixed.** A guided tour (`?tour=press` or `broadcast`) has three steps: Watch, Understand and What if?. Each step only drives the lab's own controls, so all output is the normal, labelled lab output. The guide sits above the analysis column on wide screens, so the pitch keeps its place, and above the pitch on stacked layouts. It can be exited at any time. `after/desktop-03-tour-start.jpg`, `after/desktop-05-understand.jpg`, `after/desktop-10-what-if.jpg`, `after/tablet-10-what-if.jpg`, `after/phone-10-what-if.jpg` |
| G4 | **The defender-response selector looked broken on the featured what-if.** All six policies return no simulated move when only a defender is moved: per their descriptions, they react to a moved attacker, carrier or ball. Checked on the API for `none`, `marker_follows`, `press_carrier`, `zonal_shift`, `lane_cover` and `learned`. | medium | `before/desktop-11-response.jpg` | **Fixed (explanation).** When the edit moves only defending players, "moved no defender" is followed by "You moved only defenders; the response models react to a moved attacker or ball carrier … Move an attacker to see one respond." The tour says the same. A better demonstration needs a curated attacker move (backend request 4). `after/desktop-11-response.jpg`, `after/tablet-11-response.jpg` |
| G5 | **Similar moments, events and evidence were hard to find**, far down a scrolling column. | medium | `before/desktop-08-similar.jpg` | **Fixed.** Jump links at the top of the analysis column (sticky on wide screens): Why this moment, Pass options, Similar moments, Events, Labels; in edit mode, What-if result, Defender response, Evidence & assumptions, Labels. Each focuses the target heading or control. These are buttons, not `#` links, because the hash is the router's. `after/desktop-05-understand.jpg`, `after/desktop-08-similar.jpg` |
| G6 | **A selected player's card could be off screen.** On desktop it opened at the top of an already scrolled analysis column. On stacked layouts it was below the transport and legend, with no cue. | medium | `before/desktop-09-player-card.jpg` (card not visible), `before/phone-09-player-card.jpg` | **Fixed.** On wide screens the analysis column scrolls to the card. On stacked layouts a one-line summary under the pitch ("#4 Home · Attacking third, left half-space · Player card ↓") jumps to it. `after/desktop-09-player-card.jpg`, `after/phone-09-player-card.jpg` |
| F10 | **The legend was below the fold at 1440×900**, so the control colours were unexplained. It was also about 1,250 px down at 834×1112. | medium | `before/desktop-04-observed-paused.jpg` | **Fixed.** A compact key in the pitch's bottom margin, between the attack labels ("Est. control · defending ▬ attacking", or "Δ est. control · attackers lose ▬ gain" in the difference view). It is shown when the control layer is on and paused, and hidden below 601 px, where the full legend is just under the transport. `after/desktop-05-understand.jpg`, `after/desktop-10-what-if.jpg` |
| F11 | **The player card showed the raw id (`away_17`) and centre-spot coordinates.** | low | session 3 `before/desktop-04-player-inspector.jpg` | **Fixed.** The card reads "#4 Home", then a zone line ("Attacking third, left half-space") with the position's provenance badge, then the distance to the goal the player attacks. Exact coordinates are in a "Pitch coordinates" disclosure, and in the move form in edit mode. Zone grid decision: thirds of the pitch length from the team's own direction; penalty areas named separately; five channels split at the six-yard box (±9.16 m) and the penalty area (±20.16 m), with left and right as the team attacks. Pitch size comes from the data; the box sizes are the Laws of the Game. Without a known direction the card says "direction of play unknown" rather than guessing. `after/desktop-09-player-card.jpg` |
| F12 | **Column meanings (Phys., Lane, Ctrl, Fwd; Obs., Hyp., agreement) were only in hover tooltips.** | low | session 3 `before/desktop-03-lab-paused.jpg` | **Fixed.** A visible "How to read this table" disclosure under the pass-option and delta tables. The agreement thresholds are written out (robust ≥ 90%, mixed ≥ 70%). Tooltips and screen-reader captions are kept. `after/desktop-07-evidence.jpg` |
| G7 | **No single key to the provenance language**, and the legend drew observed, hypothetical and simulated discs in the home team's amber, which read as "Home". | medium | `before/desktop-04-observed-paused.jpg`, session 3 tablet legend | **Fixed.** `ProvenanceKey` lists all five labels, each with its pitch glyph, badge, meaning and examples. It appears on the landing ("How to read what you see") and in the legend ("What do observed, derived, estimated, hypothetical and simulated mean?"), and the jump links reach it. Provenance glyphs are neutral grey; team colours stay for teams. `after/desktop-14-labels.jpg`, `after/phone-14-labels.jpg` |
| G8 | **An estimated ball looked observed.** SkillCorner extrapolates the ball off camera, and the API reports `ball_provenance: "estimated"`, but the pitch drew a solid ball. | medium (provenance) | real frame `skillcorner-1899585` f751 | **Fixed.** An estimated ball is hollow and dashed with an "est." tag, during playback (optional `ball_detected` replay mask) and on a paused frame (`ball_provenance`). The legend adds "Ball, estimated", and the frame analysis shows an "est. ball" note. The mask parse is validated and unit-tested; when the field is absent (today's backend) behaviour is unchanged. `after/desktop-22-estimated-ball.jpg` |
| G9 | **My first tour build said "Everything you see in the replay is observed tracking data" on every tour**, including broadcast, where off-camera players are estimates. | high (honesty) | `phone-15` capture during this session | **Fixed** (commit `d96e724`). Each situation words its own watch step. Broadcast: "Solid players are detected by the camera; hollow ones are off camera, and their positions are provider estimates, not observations." A unit test guards it. `after/phone-15-broadcast-tour.jpg` |
| G10 | **The live featured numbers took 8.7 s** in development. React's double effects fired each request twice, and the aborted duplicates still computed on the backend, competing with two Defensive Shadow calls on four shared cores. | low (performance) | `tools/timing.cjs` output | **Fixed.** Requests are de-duplicated (one promise per situation; failures are forgotten so Retry works) and serialised: both scenarios first, then the shadows. First numbers arrive in **1.2 s** and all of them in **2.7 s** (dev server, same machine). |
| G11 | **On a phone the guided-tour button was below the fold.** | low | `after` interim capture | **Fixed.** A full-width "Start the guided tour" button follows the lede on stacked layouts only (it is `display: none` elsewhere, so it is not a duplicate tab stop). `after/phone-01-arrive.jpg` |
| G12 | **Accessibility**: axe flagged the tour card's `<aside>` nested inside a labelled section (landmark-complementary-is-top-level). "Exit tour" was the first stop in the guide's tab order. | low | `tools/axe.cjs`, `tools/kbd.cjs` | **Fixed.** The tour card is a `<section>`. Exit is last in the DOM and positioned top right. axe reports **0 violations** in 10 states (landing with live numbers, the three tour steps, table help, labels key, player card, response and delta help; desktop and phone). Tab order: skip link → brand → data source → **Start the guided tour** → browse link → featured links; in the lab, back → mode → guide steps → guide actions → Exit → layers → Player menu → transport → legend key → jump links (`after/keyboard-order.txt`). |

Errors and empty states were rechecked. With the API stopped (a dev proxy to a dead port), the landing shows "Could not load matches" and the tour link shows "Could not open this moment", each in plain language with "Browse possessions" or Retry (`after/desktop-20-api-down-landing.jpg`, `after/phone-21-api-down-tour.jpg`). A failed featured computation is covered by an e2e test (503 mocked, then Retry). An empty possession filter offers "Clear filters".

**Console.** No errors or warnings in any journey state at any viewport. The only console errors
seen were the expected 500s from the deliberately dead proxy. **Overflow.** No horizontal page
overflow in any capture (`scrollWidth ≤ clientWidth`).

**Animation.** Only the loading skeleton's pulse and a 180 ms fade-in when the live numbers arrive.
Both are off under `prefers-reduced-motion`.

## Follow-up: a featured situation that depends on the defender response (C39)

Still-open item 1 and backend request 4 asked for a curated attacker move where a rule moves a
defender. The research search (RESEARCH.md, "Featured response example"; ledger C39) chose
Metrica Game 2, frame 31360: home #1 passes to home #5, and home #5 is moved 3 m into space.

* **Third featured card, "One run, two assumptions about the defence".** It requests the same two
  scenarios the lab makes (static defenders and `marker_follows`) and shows, for the pass that was
  played, both estimates side by side, whether the direction flips, the simulated defender move
  (`simulated` badge, with its distance from the response), the ensemble agreement under each
  assumption and the feasibility. No Defensive Shadow is requested or shown: the moved player is
  an attacker. The story says the conclusion rests on an assumption about the defence and is a
  model-based illustration, not what happened in the match. `lib/featured.ts` still holds identity
  and text only.
* **Featured facts generalised.** `moved` replaces `defender`; `showShadow` and an optional
  `response` policy are per situation. `FeaturedFacts.shadow` and `FeaturedFacts.response` are
  optional nested facts read from the API payloads.
* **Tour step 3 preselects the response** and the guide says so, labels the moved defender as
  simulated, and offers "Static vs response" (focuses the lab's "Does the defender response
  matter?" comparison) and "Static (no response)". Deep links accept `response=<policy>` (malformed
  names are ignored), so "Open the what-if directly" opens with `marker_follows` preselected too.
* **Layout.** Three cards in three columns above 1180 px, two columns plus a full-width third card
  from 1001 to 1180 px, one column at 1000 px and below. Checked at 1440×900, 834×1112 and
  390×844 (DPR 1): no horizontal overflow and no console errors or warnings on the landing or in
  the tour. The match clock in a card's kicker no longer wraps.
* **Wording to watch.** The lab's generic experimental note says response rules are "not validated
  against observed defender movement", while the story (and the policy's own assumptions) cite the
  observational check of `marker_follows` (C37). Both are true; see backend request 5.

Screenshots: `web/screenshots/landing-desktop.png` (refreshed, three cards) and
`web/screenshots/guided-tour-response-desktop.png` (new, tour step 3 with the response).

## Still open

1. ~~**The featured what-if cannot show a defender response** (G4).~~ Done; see the follow-up above.
2. On wide screens, when the page itself is scrolled to the bottom, sticky containment pushes the
   top of the analysis column (with its jump links) above the viewport. This is minor: the column
   still scrolls, and the links return when the page scrolls up.
3. Leaving the tour does not remove `tour=` from the URL, so a reload shows the guide again.
4. Carried over from session 3: the phone pitch is about 330 px wide (F17), and the lab view still
   re-renders on every frame index while scrubbing (F14).
5. The landing makes six compute requests per visit (four scenarios, then two Defensive Shadows).
   That is fine locally; for a public demo, see backend request 3.

## Backend requests (not made here; `src/` is owned by others)

1. **`ball_detected`** on the replay (in progress). The UI parses and draws it and is tested
   against the contract; nothing else is needed.
2. **Optional: policy metadata on what a response reacts to**, for example
   `reacts_to: ["attacker", "carrier", "ball"]` on `PolicyInfo`. The G4 hint currently infers
   "only defenders moved" from the scenario's `attacking_team` and player teams, and its wording
   follows the policy descriptions.
3. **Optional: cached featured results**, for example a `GET /api/featured` that returns the
   precomputed scenario and shadow summaries. The landing would not have to recompute them on each
   visit; the UI would keep reading every number from the API.
4. ~~**Research: a second featured example with an attacker move.**~~ Done (C39); shown as the
   third featured situation.
5. **Optional: say what a rule's check established.** `marker_follows` and `lane_cover` report
   `validated: false` with no `validation_ref`, although their assumptions cite the observational
   check (C37). A field such as `checks: [{kind: "observational", ref: "C37", ...}]` would let the
   UI say "checked observationally, not a forecast" instead of the generic "not validated" note.

## Tests

* Vitest: **155** (130 before). New tests cover the zone grid, featured facts (every fact read
  from synthetic payloads, nothing invented when absent), tour routes (malformed ids ignored), the
  guide bar, the featured card (loading, live, shadow unavailable, failure and retry, one request
  per situation), the player card (no raw id, an unknown direction is not guessed), the provenance
  key, the response hint, and the `ball_detected` parse contract. The broadcast tour text is tested
  never to claim everything is observed.
* Playwright: **23/23** against the real API (16 before). The new `e2e/journey.spec.ts` covers the
  landing story and live numbers, the full guided tour (with the landing's numbers equal to the
  lab's), the player card, the jump links with column help and the labels key, keyboard entry into
  the tour, and the featured-card failure and retry.
* `scripts/check.sh web`: exit code 0.
* After the response follow-up: Vitest **163** (facts generalisation, the response story's
  wording, the `response=` route, the response card and its retry, the guide's comparison step).
  Playwright **26/26**: the response card's live numbers equal the lab's under both assumptions,
  tour step 3 shows the comparison with `marker_follows` preselected, the direct link preselects it,
  plus a new screenshot test.

## Reproducing the audit

```bash
export FOOTYMIND_DATA_DIR=/path/to/data
uv run footymind serve --port 8040 &
(cd web && FOOTYMIND_API=http://127.0.0.1:8040 npx vite --port 5190 --strictPort &)
export WEB=$PWD/web
T=docs/product/audit-session4/tools
node $T/journey.cjs /tmp/after            # the nine steps through the guided tour, 3 viewports
node $T/journey-before.cjs /tmp/before    # the session-start driver (direct featured link, no tour)
node $T/kbd.cjs > /tmp/keyboard-order.txt # Tab order on the landing and in the tour
node $T/timing.cjs                        # time to the featured numbers, with the API calls
BASE=http://127.0.0.1:5190 node $T/quick.cjs /tmp/x desktop "#/lab/skillcorner-1899585%3Ap0000?f=751" ball 0
# API stopped: a second dev server proxying to a dead port
(cd web && FOOTYMIND_API=http://127.0.0.1:8099 npx vite --port 5191 --strictPort &)
BASE=http://127.0.0.1:5191 node $T/quick.cjs /tmp/x desktop "" api-down 0
# axe: install axe-core outside the repository and point the script at it
npm i --prefix /tmp/axe axe-core@4.10.3
AXE_JS=/tmp/axe/node_modules/axe-core/axe.min.js node $T/axe.cjs
```
