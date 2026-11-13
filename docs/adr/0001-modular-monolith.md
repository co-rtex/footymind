# ADR-0001: Modular monolith, local files, no service infrastructure

- Status: accepted (2026-11-13)

## Context
The workload is a few public matches (about 24 MB each in memory), single-user interactive
analysis, and offline experiments. The latency-critical path (frame analysis, about 70 ms) is
numpy-bound in one process.

## Decision
One Python package and one FastAPI process. Processed data lives in versioned local bundles
(npz + Parquet + JSON manifest) loaded into memory with an LRU cache. There is no Postgres,
Redis, queue or microservice. The built frontend is served by the same process.

## Consequences
+ Trivial to run, test and deploy. Reproducibility comes from manifests and pinned hashes.
− Horizontal scaling and multi-user write workloads are out of scope. Revisit if hosted
  usage requires persistence of user scenarios (then consider SQLite before Postgres).
