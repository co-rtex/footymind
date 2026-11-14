# FootyMind 1.0: final product QA (2026-11-14)

Scope: the web lab (`web/`) against the real API and all 23 processed matches (pipeline v3,
both similarity indexes). The release is feature-frozen, so only demonstrated usability,
accessibility, responsiveness, error-handling and visual defects were fixed. Every check ran in
Chromium through Playwright (scripts in [`tools/`](tools/)). Screenshots are in
[`screenshots/`](screenshots/).

## What was exercised

The first-visit journey ran at **1440×900, 820×1180, 1180×820 and 390×844**: landing, guided
tour, each featured situation, watch, pause, analysis overlay and *Why this moment*, pass options,
similar moments, the curated move, Observed / Hypothetical / Difference, **all 7 response
policies**, evidence and caveats, then back to match selection and a new analysis (a SkillCorner
match). The featured situations were `metrica-sample-1` f53708, `skillcorner-1899585` f39177 and
`metrica-sample-2` f31360. A defender was also moved with the keyboard move form.

* Each featured situation opened at its own frame (analysis, *Why* and the URL all agree) and
  showed the right match title and source. Off-camera players were labelled *estimated* on the
  broadcast situation.
* Results: 4 viewports × 3 situations × 7 policies, and every scenario request succeeded.
  There were **0 console errors and 0 failed requests**, and no horizontal overflow at any of
  the four viewports.
* Error and empty states:
  * An unknown match, an unknown possession and an out-of-range frame were tried as deep links.
  * Bogus `response=`, `tour=` and `move=` parameters were tried as well.
  * The API was stopped mid-session (`api-down-*.jpg`). Analysis, similar moments, the what-if
    and the landing each show a plain-language alert with Retry.
  * The SkillCorner similarity index was removed in a scratch copy of the data. The panel says
    the index has not been built, and Metrica still works (`similar-index-missing.jpg`).
* Keyboard only, with `prefers-reduced-motion: reduce`: landing → tour → play/pause → steps → jump
  links → *Why* statements → step 3 → every policy with arrow keys → Player menu → move form → back.
  There were no traps, and focus stayed visible (the scrubber uses the thumb ring in Chromium).
  The only animation is user-started playback; the spinner and skeletons already honour reduced
  motion.
* axe-core 4.10.3, installed outside the repository, checked 8 states per viewport (desktop,
  tablet and phone): the landing; steps 2 and 3 of each situation with similar moments, table
  help and `learned_innovation`; and an unknown possession.

## Findings

| # | Severity | Finding | Status |
|---|---|---|---|
| Q1 | Medium | **Back from the lab** ("← Possessions" or the browser's Back) opened the landing at the top. The possession list the visitor came from starts 2,291 px down on desktop and 5,910 px down on a phone, and focus was lost to `<body>`. | **Fixed.** The list is scrolled into view and its heading focused once it loads (`after-phone-back-to-possessions.jpg`). A card click afterwards does not jump or take focus. |
| Q2 | Medium (a11y) | **The match cards were `role="radio"` buttons, yet each was a Tab stop and arrow keys did nothing.** That made 23 Tab stops between the featured links and the possession list, against the radio-group pattern. | **Fixed.** One roving Tab stop. Arrow keys, Home and End move the selection across both provider groups. |
| Q3 | Medium | **Tablet landscape (1180×820)**: the full-width pitch was 1130×757. After *Watch* scrolled it to the top, the play/pause control was below the screen (`before-tablet-landscape-watching.jpg`). | **Fixed.** The canvas width also respects the viewport height and is centred: 973×652, with the transport visible (`after-tablet-landscape-watching.jpg`). Desktop, tablet-portrait and phone sizes are unchanged. |
| Q4 | Low (a11y) | **Full-page loading and error states had no `h1`.** axe reported `page-has-heading-one` at all 3 viewports. | **Fixed.** The status title is the page heading there. axe now reports **0 violations in 24 states**. |
| Q5 | Low | `document.title` stayed "FootyMind · The Invisible Game" on every `#/at/` link (all featured and tour links). | **Fixed.** These links are now titled "Lab · FootyMind". |
| Q6 | Low | **An unknown match was reported as a missing frame**: "We could not find a possession at frame 100 of this match". Its *Browse possessions* button, and any `#/match/<unknown>` link, showed the landing with no list and no explanation. | **Fixed.** The alert reads "We could not find the match "…" on this server". The landing says "Match not found" where the list would be (`after-unknown-match-*.jpg`). |
| Q7 | Low (WCAG 1.4.10) | **At 320 px, edit mode scrolled sideways** (375 px document): long policy names in the select, a nowrap segmented control, the summary table, and an unbreakable evidence path in the `learned_innovation` note. The landing was 330 px wide (the response card's table), and at 390 px the scenario panel overflowed by 2 px. | **Fixed** (`before/after-320-edit-full.jpg`), with a Playwright test. |
| Q8 | Low | When the what-if request failed, the pitch badge still said "Edit scenario: drag a player" with a move applied (`api-down-what-if.jpg`). | **Fixed.** The badge says "What-if failed: see the scenario panel". |
| Q9 | Low | The event ticks on the scrubber are 10×12 px tap targets. | Open. The Events list offers the same jumps (WCAG 2.5.8 equivalent-control exception). A larger hit area is suggested after 1.0. |
| Q10 | Low | On short wide screens (1280×720) the transport sits below the fold, because the wide layout does not scroll the pitch. | Open. 1440×900 is fine. |
| Q11 | Low | Some small targets are under 24 px: the "Ends in a shot" label is 19 px tall and the team-chip links in the possession table are 23 px tall. | Open. The whole table row is clickable. |
| Q12 | Info | Pass-probability labels on the canvas can overlap player markers in crowded frames (`desktop-04-response-what-if.jpg`, "93% (−" under #6). | Open. This is renderer polish, not a defect under the freeze. |
| Q13 | Info | Moving a defender, as the press and broadcast situations do, makes every response policy move nobody. | Known, and explained in the panel and the tour (session 4, G4). |
| Q14 | Info | A hand-edited unknown `response=` adds an "(experimental)" option and shows the API's 422 explanation. Bogus `tour=` and `move=` are ignored. | Acceptable: the backend validates the policy. |
| Q15 | Info | Entering the lab leaves focus on `<body>`, so the next Tab reaches the skip link. Through the Vite proxy, a stopped API shows "HTTP 500" in the small technical line; the plain text is still right. | Acceptable. |
| Q16 | Info | The scrubber's focus ring is styled only for the WebKit thumb. | Not verified: Chromium was the only browser available. |

## Tests

* `cd web && npm run typecheck && npm run lint && npm test && npm run build` all passed.
  **222 vitest** (213 before). The new tests cover `pitchCssWidth`, the page-level
  `StatusBlock` heading, the keyboard radio group, the unknown-match notice and the
  return-to-list reveal.
* `FOOTYMIND_DATA_DIR=<data> npx playwright test --trace off` (its own servers on 8765/5173):
  **31/31 passed**. The 4 new tests cover back to the list (button and browser Back), titles
  and the unknown match, no sideways scroll at 320 px, and the tablet-landscape transport in view.
  Screenshot churn from that run was not committed.
* axe: 3 violations before, **0 after** (24 states).
