# JBang SDK Examples Migration Status

Date: 2026-05-13  
Status: In Progress (compatibility-first)

## What Changed

- Index and quickstart docs now point to verified runnable scripts.
- Canonical trainer lane is now documented at `../trainer/`.
- Older `gollek-sdk-*` examples are now runnable compatibility samples.

## Why This Report Replaced the Old One

The previous report described a fixed v0.2 snapshot with file names and claims
that no longer match current repository structure. This replacement tracks the
actual migration state instead of declaring the lane "complete."

## Current Reality

- Useful and runnable examples exist across `sdk/`, `trainer/`, `nlp/`, and
  `multimodal/`.
- Artifact naming is still mixed between canonical (`gollek-ml-*`,
  `gollek-trainer-*`) and compatibility (`gollek-sdk-*`) surfaces.
- Some examples still use local jar paths under `${user.home}/.gollek/jbang/libs`.

## Next Cleanup Targets

1. Move remaining compatibility scripts to canonical dependency coordinates.
2. Add one canonical example each for:
   - trainer runtime
   - model preparation and provider selection
   - SDK feature-kit flows
3. Keep legacy examples runnable, but label them clearly as transitional.
