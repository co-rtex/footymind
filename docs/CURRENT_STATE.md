# Current state and session handoff

_Last updated: 2026-11-13 23:39 UTC, end of session 2 (21:34–23:39 UTC). Keep this file
factual and update it at every checkpoint._

## Branch and repository
* Working branch: `claude/loving-wright-i0shmp` (all commits pushed to `origin`). `main` is
  untouched, and no PR has been opened (the owner has not asked for one).
* Commits are authored as Christian Cortez with no assistant trailers (see CLAUDE.md).
* Only the main worktree remains. The merged specialist and backend worktrees and their local
  branches have been removed; only `main` and this branch exist locally.
* No background processes are left running (dockerd, uvicorn and vite were all stopped).
* **Stop-hook note.** The cloud environment runs a *managed* Stop hook
  (`~/.claude/stop-hook-git-check.sh`, from the environment's launcher settings, not this repo,
  and protected from edits). It asks for commits to be re-authored as "Claude". That conflicts
  with the owner's authorship rule and is deliberately not done. "Unverified" only means the
  commits are unsigned. Options for the owner: change the environment hook, or enable commit
  signing. Do not rewrite history.

## Data (local, not committed)
`data/raw` holds Metrica (about 180 MB) and SkillCorner (about 1.8 GB). `data/processed` holds
23 bundles at **pipeline version 2**. To rebuild from scratch:
```bash
uv run footymind download && uv run footymind ingest   # all registered sources (~2 GB)
uv run footymind download metrica-sample-1 metrica-sample-2 metrica-sample-3   # Metrica only
```

## Verification (last run in this session)
| Gate | Result |
|---|---|
| `scripts/check.sh` (exit code 0) | Python: ruff, format, mypy, **78 pytest**; web: typecheck, lint, **74 vitest**, build |
| `cd web && npx playwright test e2e/flow.spec.ts` | **6/6** against the real API (Metrica + SkillCorner, featured scenarios, shadow layer, broadcast provenance) |
| Docker | image built locally, e2e passed against the container; CI `docker` job green |
| GitHub CI | green through 52f9ed2 (run 25). Runs 22 and 24 were cancelled by newer pushes (concurrency), not failed. Runs for 97f6ef0 and later were in progress at handoff |

## Research status (numbers live in CLAIMS_LEDGER.md)
* **Main, replicated result:** decision-time spatial context improves held-out pass-outcome
  prediction. Metrica −0.026 [−0.034, −0.018] (C20). SkillCorner −0.021 [−0.025, −0.018],
  better in 20/20 held-out matches (C28; adapter v2 after the second independent review).
  The drivers are defenders goal-side of the ball and attacking width (C26).
* Action-conditioned gain: −0.071 / −0.046. This is provider-dependent and partly execution quality.
* Interaction features: null on 3 matches, −0.004 on 20 matches. This is not a case for GNNs yet.
* Physics: literature parameters are miscalibrated, and fitted parameters hit implausible bounds,
  so the model is structurally misspecified (C7, C15). Per-provider option models serve
  probabilities (C21, C24).
* Defensive Shadow: robust to noise, 2× chance at the interceptor, but **not better than
  nearest-defender** (C16, C17). On SkillCorner the shadow is at chance and worse than
  nearest-defender (C29): the outcome signal does not replicate. Run value: tiny association and at-chance ranking on both
  providers (C22, C25). Both are experimental.
* Two independent reviews were done this session. Their findings are fixed: rules v2,
  decision-time framing, scenario feasibility and ensemble semantics, Shadow framing, and
  the SkillCorner adapter v2.

## Next executable actions
1. Confirm that CI is green for the last commits (97f6ef0 onwards) with `mcp__github__actions_list`.
   Optionally run shadow validation on all 20 SkillCorner matches
   (`python -m footymind.research.shadow_validation skillcorner`, about 18 min).
2. Research: simulated defender responses (learned ghosting) before any off-ball valuation
   claim. Add time-on-ball / one-touch context to both providers' baselines and re-run C20/C28
   (the second review estimated −0.018 for SkillCorner with it).
3. Product: done. Stable deep links `#/at/<match>?f=<frame>&move=...` resolve the possession
   server-side (`GET /api/matches/{id}/frames/{frame}/possession`), and the featured scenarios
   use them. Possession-id links (`#/lab/<pid>`) still work but can change between pipeline
   versions.
4. Hygiene: Vitest 5 upgrade (dev-only audit findings).
5. Reproduction: `scripts/reproduce.sh --dry-run` lists every committed research artefact in
   dependency order. Run it from a clean `src/` (it refuses otherwise).

## Decisions needing the owner
* Whether to host a public demo serving provider-derived positions (Metrica has no licence
  file; SkillCorner is MIT with a credit request).
* Whether to open a PR from this branch to `main`.
* Whether to change the managed stop hook or enable commit signing.
