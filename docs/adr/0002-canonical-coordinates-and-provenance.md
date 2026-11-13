# ADR-0002: Canonical coordinates and provenance labels

- Status: accepted (2026-11-13)

## Decision
* Canonical metres, centre origin, y up, provider orientation retained, with per-period
  attacking direction stored explicitly (see ARCHITECTURE.md#coordinates).
* Every analytical quantity carries `Provenance`: observed, derived, estimated,
  hypothetical or simulated. Unobserved values are NaN/null and never filled silently.

## Rationale
Flipping coordinates at ingestion would hide the half-time switch and make
provider-frame validation harder. Keeping the provider orientation plus an explicit
transform keeps the raw-to-canonical mapping a fixed affine map that can be tested
against provider rows. Provenance labels are how the product stays scientifically
defensible when observed and hypothetical states appear side by side.
