# Data licences and distribution

What the two data providers allow, what FootyMind does with their data, and which question is
left for the owner. Checked on 2026-11-14 against the providers' repositories (quoted below).
This is a record of the published terms, not legal advice.

## The terms as published

| | Metrica Sports sample data | SkillCorner open data |
|---|---|---|
| Source | https://github.com/metrica-sports/sample-data | https://github.com/SkillCorner/opendata |
| Licence file | **None** | **MIT** (`LICENSE`, "Copyright (c) 2020 SkillCorner") |
| Stated purpose | "to allow potential clients to check out this data [...]" and "to provide some sample data to play around with to the broader football analytics community" | to let prospects "access data easily, parse our data format and get started building on top of it" |
| Stated conditions | "Please be responsible with the use of this data." "If you use it for anything public, please aknowledge the source." | "if you use the data, we kindly ask that you credit SkillCorner" (and notify them) |
| Identities | anonymised (no player, team or competition names) | real team names, competition and date; FootyMind shows team short names, never player names |

## What FootyMind does

* **No raw data is redistributed.** `data/` is git-ignored and `.dockerignore`d; files are
  downloaded from the providers' own repositories at setup and verified against pinned
  SHA-256 checksums (`src/footymind/data/sources.py`). The Docker image contains no match data;
  the container downloads it into its volume on first start.
* **Committed artefacts are derived.** `experiments/results/*/results.json` hold aggregate
  metrics and a few example frame references; the screenshots in `web/screenshots/` and
  `docs/product/` show rendered views of the data (anonymised for Metrica).
* **Attribution.** The README, `docs/DATA_SOURCES.md`, the lab's landing page (provider notes
  with links) and every match view (`source-attribution`) name the source; SkillCorner is
  credited by name.

## Deployment configurations

| Configuration | Sources | Basis |
|---|---|---|
| Local / research (default) | Metrica 1–3, optionally all 20 SkillCorner matches | both providers' stated purposes (analysis, community use) with attribution |
| **Public demo, licence-clear** | SkillCorner only, e.g. `FOOTYMIND_SOURCES="skillcorner-1874553 skillcorner-1899585 skillcorner-2017461"` (the broadcast featured situation's match included) or `FOOTYMIND_SOURCES=skillcorner` | MIT plus the credit request, which the UI satisfies |
| Public demo with Metrica | add `metrica` | **owner decision**: Metrica asks for responsible use and acknowledgement for public use but publishes no licence |

The SkillCorner-only configuration was verified on 2026-11-14 through the real entrypoint
(`docker/entrypoint.sh`, local process mode): download with checksums, ingest, similarity
index, `scripts/smoke_api.py --require-similar` passed, and the lab's landing page loaded with
no errors, the broadcast guided tour and featured card, and a note that the two Metrica
featured situations need data not loaded on that server.

## Open question (owner)

Whether hosting a public demo that renders Metrica's sample data is within "be responsible
with the use of this data" with acknowledgement. Nothing in the repository depends on the
answer: choose the SkillCorner-only configuration to avoid the question, or ask Metrica.
