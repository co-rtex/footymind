# FootyMind web lab

The browser front end for FootyMind 2.0 ("The Invisible Game"). It replays real
tracking data, overlays the spatial model's estimates on a paused frame, and lets
you edit a frame to compare observed and hypothetical model outputs.

![Possession lab with estimated pitch control and ranked pass options](screenshots/lab-analysis-desktop.png)

| Scenario, difference view | Phone width |
| --- | --- |
| ![Edited scenario: moved player, simulated marker response and control delta](screenshots/scenario-difference-desktop.png) | ![Lab stacked at 390 px](screenshots/lab-narrow.png) |

![Landing page with match selector and possession list](screenshots/landing-desktop.png)

Data: Metrica Sports sample games (real, anonymised tracking at 25 Hz with
synchronised events), https://github.com/metrica-sports/sample-data. The app
shows the attribution returned by the API on every page.

## Run

```bash
# 1. backend (from the repository root)
uv sync
FOOTYMIND_DATA_DIR=/path/to/data uv run uvicorn footymind.api.app:app --port 8765

# 2. frontend dev server (proxies /api to :8765)
cd web
npm install
npm run dev            # http://localhost:5173
```

Production: `npm run build` writes `web/dist`, which FastAPI mounts at `/` when it
exists, so `http://127.0.0.1:8765/` serves the app and API from one origin.
Set `FOOTYMIND_API` to proxy the dev server to a different backend.

## Check

```bash
npm run typecheck   # tsc, strict
npm run lint        # eslint, zero warnings
npm test            # vitest unit and component tests (synthetic fixtures)
npm run build
npm run e2e         # Playwright against the real backend and real data
npm run screenshots # regenerate screenshots/ at 2x (SHOT_DPR, SHOT_DIR override)
```

`npm run e2e` starts the backend (`uv run uvicorn ...` from the repository root)
and the Vite server unless they are already running, so export
`FOOTYMIND_DATA_DIR` first. Set `E2E_BASE_URL=http://127.0.0.1:8765` to test the
production build served by FastAPI instead. With a preinstalled Chromium the
config falls back to `/opt/pw-browsers/chromium`, or you can set `PW_CHROMIUM`.

## Using the lab

1. Pick a match and open a possession (team, period, match clock, duration,
   passes, and shot or goal badges).
2. Replay it: <kbd>Space</kbd> plays or pauses, <kbd>←</kbd>/<kbd>→</kbd> step a frame
   (<kbd>Shift</kbd> steps 1 s), 0.5×/1×/2× speed, and the timeline has event
   markers. Trails show the last 1.5 s.
3. Pause on any frame to see estimated pitch control, pass options (line width and
   opacity encode the option-model probability) and pressure on the carrier. The
   side table ranks the options and also serves as the text alternative.
4. **Edit scenario**: drag up to four players. While you drag, the dashed circle
   shows where the player could have reached (7 m/s from the observed position
   1.5 s earlier). On drop, the backend re-runs the model. You can choose the
   opponent response (none or marker follows), compare by the option model or the
   physics heuristic, switch between observed, hypothetical and difference views,
   and read the deltas with their 10–90% ensemble ranges, sign agreement and the
   evidence and assumptions. **Reset to observed** returns to the data.
5. Click a player to inspect id, team, position, speed and provenance. In edit
   mode the player card also offers a keyboard-accessible move form.

## Provenance on screen

| Label | Pitch | Text |
| --- | --- | --- |
| observed | solid disc | solid grey badge |
| derived | n/a | outlined badge |
| estimated | translucent heatmap and pass lines, "model" labels | dotted amber badge |
| hypothetical (you moved) | translucent disc, dashed violet ring, ghost and dashed arrow | dashed violet badge |
| simulated (rule response) | translucent disc, dotted white ring, dotted arrow | dotted white badge |

Pitch control is always labelled "Estimated pitch control (model, uncalibrated)".
Pass probabilities name their model: the calibrated option model is primary, and
the physics heuristic is shown as uncalibrated. API warnings appear verbatim.

## Architecture

```
src/api/        types.ts mirrors src/footymind/api/schemas.py; parse.ts validates
                payloads and turns the replay into typed arrays (NaN = not observed);
                client.ts is the typed fetch client with readable FastAPI errors
src/lib/        coords (metres <-> canvas, y flip), interp (sub-frame interpolation
                that never draws an unobserved position), playback (rAF clock),
                formatting, routing, model labels
src/render/     PitchRenderer: imperative Canvas 2D, HiDPI, cached markings and
                rasters; theme colours read from the CSS tokens
src/components/ React UI; the Lab orchestrates playback, analysis and scenarios
src/styles/     tokens.css (design tokens as CSS variables) and app.css
e2e/            Playwright flow test and screenshot generator
```

React never re-renders per animation frame. The playback clock writes `t` into
the renderer and updates the clock and scrubber DOM nodes directly. Analysis is
fetched only when paused, debounced while scrubbing, and cached per frame.
