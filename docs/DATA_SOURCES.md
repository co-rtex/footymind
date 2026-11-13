# Data sources

Raw data is **never committed**. `footymind download` fetches the registered files and
verifies them against SHA-256 checksums pinned in `src/footymind/data/sources.py`.
`footymind ingest` builds processed bundles. Every bundle's manifest records the input
hashes and code revision.

## In use

### Metrica Sports sample data (Sample Games 1 and 2)

| Field | Value |
|---|---|
| URL | https://github.com/metrica-sports/sample-data (files under `data/`, `master` branch) |
| Content | 2 full anonymised matches: player and ball tracking at 25 Hz (optical; all players on the pitch), plus synchronised manually annotated events |
| Pitch | 105 × 68 m. Coordinates are normalised to [0, 1], with (0,0) at the top-left |
| Licence | **No licence file.** The README asks users to "be responsible with the use of this data" and to acknowledge the source if it is used publicly. FootyMind does not redistribute raw files, and the UI and README attribute the source. |
| Pinned files | `Sample_Game_{1,2}_RawEventsData.csv`, `..._RawTrackingData_{Home,Away}_Team.csv` (hashes in `sources.py`) |
| Verified facts | 145,006 / 141,156 frames. Ball observed in 60.9% / 59% of frames (missing mostly during dead ball). Event/tracking sync: median passer–ball distance at pass start 0.084 m / 0.088 m. Receiver–ball at pass end 0.22 m |
| Known quirks | Game 2 labels one player `"Player 26"` (with a space) in both files (handled, regression-tested). Substitutes are NaN while off the pitch. Rare tracking artefacts: 20 m/s linear drifts and snap-backs, and teleports up to 91 m (flagged by `implausible_motion_mask`, about 0.01% of player-frames). A few pass start annotations lag the kick, with the ball already 2–4 m away (handled by the event-based carrier fallback). |
| Teams/players | Anonymised (`Home`/`Away`, `PlayerN`). The UI must not invent names. |

### Metrica Sample Game 3 (registered, not yet ingested)

EPTS/FIFA format: `tracking.txt` + `metadata.xml` + JSON events. Hashes are pinned. An
adapter is planned (see ROADMAP). Adding it gives a third leave-one-match-out fold.

## Event semantics used for labels (Metrica)

* `PASS` is a completed pass, with a receiver (`To`).
* Failed pass attempts appear as `BALL LOST` with subtype null, `INTERCEPTION`, `HEAD`,
  `CROSS-INTERCEPTION`, `HEAD-INTERCEPTION`, `GOAL KICK[-INTERCEPTION]` or `DEEP BALL`, and
  optionally as `BALL OUT` with subtype null, `HEAD`, `CROSS` or `GOAL KICK`.
* Excluded, as not pass attempts or not pass outcomes: `THEFT`, `FORCED`, `HAND BALL`,
  `OFFSIDE`, `END HALF`, `REFEREE HIT`, `WOODWORK` and clearances.
* **For failed passes, `End X/Y` is the location where the ball was lost, not the intended
  target.** It is therefore excluded from all features (label leakage).

## Possession definition (v1)

A possession is a maximal run of on-ball events credited to one team within a period. It
ends on a change of team, a period change, or after `BALL OUT`/`SHOT`. A `SET PIECE` always
opens a new possession. `CHALLENGE`, `CARD` and `FAULT RECEIVED` never switch possession.
Game 1: 381 possessions. Game 2: 366.

## Candidates investigated, not yet integrated

| Source | Notes | Status |
|---|---|---|
| SkillCorner open data (https://github.com/SkillCorner/opendata) | Broadcast tracking (players out of camera view are missing or extrapolated, so visibility must be modelled). Newer releases include dynamic events (off-ball runs, passing options). | Not yet verified in this session. Licence and schema must be checked before use. |
| StatsBomb / Hudl open data (https://github.com/hudl/open-data) | Large event dataset. 360 freeze-frames are **snapshots at events, not continuous tracking**, and show only the players visible in the broadcast frame. | Candidate for an event-scale xT/possession-value surface. Licence terms must be followed (attribution, non-commercial). |
| kloppy (https://github.com/PySport/kloppy) | Provider-agnostic loaders | Candidate for the EPTS Game 3 adapter and future providers. Cross-validate against our adapter on Games 1–2. |

Datasets from different providers are **never pooled as one population** without explicit
provider/competition indicators and separate evaluation.
