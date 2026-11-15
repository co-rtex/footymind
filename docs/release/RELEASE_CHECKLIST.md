# FootyMind 1.0 release checklist

Status of each release criterion at sign-off (2026-11-14). **Verified** means checked by a
test, a run or an inspection recorded here or in [`VERIFICATION.md`](VERIFICATION.md);
**owner** means technically ready and waiting on an explicit owner decision.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | All core application features work | Verified | Playwright 31/31 on all 23 matches; browser QA over 4 viewports × 3 featured situations × 7 response policies with 0 console errors and 0 failed requests ([QA report](../product/release-1.0/QA_REPORT.md)) |
| 2 | The codebase builds cleanly | Verified | `scripts/check.sh` exit 0: ruff, format, mypy (`src`, `tests`, `scripts`), web typecheck, lint, production build |
| 3 | The complete test suite passes | Verified | 355 pytest passed (0 skipped, all data present), 222 vitest, Playwright 31/31 |
| 4 | All featured demonstrations work | Verified | each of the three opens at its frame with the right match and provenance (QA report; `journey.spec.ts`); hidden with a note when its data is not loaded |
| 5 | The Docker application starts | Verified | CI `docker` job on every push: build, first start on an empty volume, `smoke_api.py --require-similar`, `HEALTHCHECK` healthy |
| 6 | Existing-volume recovery verified | Verified | `scripts/container_recovery.py docker` in the built image in CI (healthy restart, outdated bundle, interrupted first start, unknown source, damaged file, restart on a damaged volume); `local` mode 49/49 |
| 7 | Fresh setup instructions work | Verified | fresh clone, empty data dir, README minimal path as written; a test checks every documented download/ingest command |
| 8 | Research claims are traceable | Verified | every cited claim id exists and every ledger results file is committed with its revision (`tests/test_claims_ledger.py`); two independent numbers audits; C20 re-run reproduces exactly |
| 9 | No critical or high-severity bugs remain | Verified | final independent review: no critical or high findings; its 2 medium and 7 low findings are fixed (ae9188c) |
| 10 | Public-facing documentation complete | Verified | README, research report, model cards, claims ledger, project summary with demo walkthrough, release notes, deployment guide, data licences, roadmap |
| 11 | Deployment configuration ready | Verified | Dockerfile, entrypoint, `DEPLOYMENT.md` (variables, sizing, health, upgrade, rollback, repair); licence-clear SkillCorner-only configuration verified |
| 12 | The repository is clean | Verified | no tracked data, caches, pickles or secrets; no broken relative links; worktrees removed |
| 13 | Final code pushed to the intended branch | Verified | `claude/loving-wright-i0shmp` |
| 14 | CI green on that revision | Verified | see "CI" below |
| 15 | Release PR prepared, not merged | Verified | non-merging PR from `claude/loving-wright-i0shmp` to `main` |
| 16 | External approvals identified | Owner | merge and tag; public deployment and Metrica's terms for a hosted demo; a code licence; stop-hook / commit-signing policy |

## CI

Every required job (Python lint/types/tests, real-data tests with the Metrica browser suite,
legacy, web, Docker runtime with entrypoint recovery) must succeed on the final release-candidate
commit. The run for the last code change is recorded below; the sign-off report names the run
for the final commit.

## Known, accepted limitations (not release blockers)

Listed in [`RELEASE_NOTES_1.0.md`](RELEASE_NOTES_1.0.md) and the QA report (Q9–Q16: small tap
targets on the scrubber ticks, the play control below the fold on a 1280×720 desktop, and
informational items).
