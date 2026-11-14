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

### SkillCorner open data (A-League 2024/25, 20 matches), integrated 2026-11-13

| Field | Value |
|---|---|
| URL | https://github.com/SkillCorner/opendata (tracking via Git LFS media URLs) |
| Licence | MIT. The provider asks users to credit SkillCorner. Raw files are not redistributed. |
| Content | 20 matches: `match.json` (lineups, pitch size, attacking side per period, period frame ranges), broadcast tracking at **10 fps** (`tracking_extrapolated.jsonl`), dynamic events (player possessions with pass outcome, passing options, off-ball runs) |
| Pinned files | `src/footymind/data/skillcorner_manifest.json` (SHA-256 for 60 files, about 1.8 GB) |
| Visibility | Players outside the camera view carry provider **extrapolations** (`is_detected = false`): only about 22% of present player-frames are detected. FootyMind stores a `detected` mask and labels extrapolated positions `estimated`. About a third of in-period frames have no player data at all (replays or close-ups) and stay NaN. |
| Verified facts | The declared home attacking side agrees with goalkeeper-based inference in all 20 matches. The median passer-to-ball distance at the release frame is 1.12–1.39 m: a 10 fps release catches the ball leaving the foot, so the sync gate is 2.5 m for this provider. The passer is detected at about 95% of releases. Pitch lengths are 104–106 m, and goal geometry uses each match's length. |
| Label mapping (adapter v2, pipeline 2) | `player_possession` with `end_type = pass`: `successful` → completed (receiver = `player_targeted_id`); `unsuccessful` → failed, recorded as `BALL OUT` when play stopped for a throw-in, corner or goal kick (about 18% of failures), `BALL LOST` with no subtype for a free kick, otherwise `INTERCEPTION`. Offside is skipped (no null outcomes exist). Context: restarts awarded to the team (`game_interruption_before = *_for`) emit `SET PIECE` in the release frame. Possessions starting with a recovery or interception emit `RECOVERY`. Release frame = possession `frame_end`, the last frame the passer controls the ball; the release window is 2 frames at 10 fps. |
| Caveats | Provider-specific: event definitions, frame rate and visibility differ from Metrica, so results are **evaluated separately and never pooled**. There are no duel events (`CHALLENGE` is never produced), and the restart mapping is approximate. Event positions (`x_end`/`y_end`, attacking-team frame) rotated by the attack sign coincide with the tracked passer in all 20 matches. That independently verifies the x and y conventions. Player names exist in the metadata but are deliberately not stored. |

## Known data-quality limitations (measured; `experiments/results/data_quality_v1`, C30)

* **SkillCorner event start frames often already contain the kick.** The ball is already moving
  away from the passer at the start frame in 62.7% of passes (Metrica 1.8%). Decision-time
  research therefore reads a guard frame 0.2 s earlier (ADR-0007, C38).
* **Broadcast visibility.** 66.6% of on-pitch SkillCorner positions at pass starts are
  camera-detected (87.4% within 25 m of the ball). Player and ball extrapolations are labelled
  *estimated* (re-ingest bundles from before `ac0629f` to get the ball mask).
* **Possession boundaries.**
  * 17 of 4,872 SkillCorner possessions (33 passes) end in a different tracking period than
    they start, because event and tracking period boundaries disagree.
  * 63 SkillCorner possessions last longer than 120 s, because long stoppages are not split.
    This inflates `time_in_possession_s` for the passes in them.
  * On Metrica, 426 of 1,222 possessions overlap the next one, by a median of about 0.1 s and
    at most 6 s. This is the lost pass's arrival frame against the recovery.

  These are documented rather than re-segmented. Re-segmenting would change possession ids and
  every committed result for well under 1% of passes. It is a candidate for a future pipeline
  version.
* A handful of passes have no ball or no passer position at the start frame (4 + 4 on
  SkillCorner). Their features stay NaN and are never filled.

## Candidates investigated, not yet integrated

| Source | Notes | Status |
|---|---|---|
| StatsBomb / Hudl open data (https://github.com/hudl/open-data) | Large event dataset. 360 freeze-frames are **snapshots at events, not continuous tracking**, and show only the players visible in the broadcast frame. | Candidate for an event-scale xT/possession-value surface. Licence terms must be followed (attribution, non-commercial). |
| kloppy (https://github.com/PySport/kloppy) | Provider-agnostic loaders | Candidate for the EPTS Game 3 adapter and future providers. Cross-validate against our adapter on Games 1–2. |

Datasets from different providers are **never pooled as one population** without explicit
provider/competition indicators and separate evaluation.
