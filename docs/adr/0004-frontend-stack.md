# ADR-0004: Vite + React + TypeScript with a Canvas 2D pitch renderer

- Status: accepted (2026-11-13)

## Context
The UI is a single interactive lab page: 25 Hz playback of 22 players plus a ball, overlays
(control grid, lanes) and drag-to-edit, backed by a Python API that owns all analytics.

## Decision
Vite + React 18 + strict TypeScript. The pitch is drawn imperatively on Canvas 2D from refs
(no React re-render per animation frame). There is no SSR framework. The production build
is static and served by FastAPI.

## Rationale
Next.js SSR/routing adds no value for a canvas-centric single page whose data comes from a
separate API. Vite gives faster builds and a simpler static deploy. Analytics stay in
Python (one source of truth) rather than being duplicated in TypeScript.
