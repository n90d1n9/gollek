# Generative Recursive Reasoning Integration Plan

This note maps `arXiv:2605.19376` into Gollek's current codebase.

Paper citation:

- Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, and Sungjin
  Ahn. "Generative Recursive Reasoning." arXiv:2605.19376, 2026.
  DOI: `10.48550/arXiv.2605.19376`.

## Architectural conclusion

GRAM should be treated as a new recursive-reasoning family, not as:

- an enhancement to `training/gollek-train-diffusion-opd`
- an extension of the existing byte-latent language-model family
- a feature hidden inside the old preprocessing-oriented
  `training/gollek-train-transformer`

Why:

- GRAM's core abstraction is a stochastic latent reasoning trajectory
- it adds width-based inference scaling through parallel trajectory sampling
- it relies on amortized variational training rather than standard deterministic
  next-step refinement alone

That puts it closer to a dedicated reasoning/model-family lane than to
diffusion reporting, byte-level language modeling, or preprocessing utilities.

## Best-fit module direction

Recommended additions and follow-ups:

- `ml:gollek-ml-reasoning-core`
  - shared latent-state, trajectory, and reasoning-score contracts
- `ml:gollek-ml-recursive-reasoning`
  - GRAM-style stochastic transition families, latent trajectory sampling, and
    recursive reasoning model metadata
- `trainer:gollek-trainer-recursive-reasoning`
  - recursive reasoning session/config/runtime support built on Gollek's
    canonical trainer seams
- `examples:gollek-ml-examples`
  - structured reasoning probes and compact report examples

## Reuse points in the current repo

These are the existing seams GRAM-style work should reuse:

- tensor/runtime foundations:
  - `gollek/core/gollek-tensor`
  - `gollek/ml/gollek-ml-autograd`
- trainer/runtime surfaces:
  - `gollek/trainer/gollek-trainer`
  - `gollek/trainer/gollek-trainer-api`
  - `gollek/training/gollek-train-data`
- benchmark/report precedents:
  - `gollek/ml/gollek-ml-byte-latent`
  - `gollek/training/gollek-train-diffusion-opd`

The byte-latent family is a particularly useful precedent for how Gollek can
host a paper-specific model family without mixing it into diffusion-specific
training packages.

## What should not happen

Do not land GRAM as:

- a diffusion OPD subfeature
- a byte-latent model variant
- a preprocessing helper under `gollek-train-transformer`

Those placements would blur the architectural boundary between:

- recursive probabilistic reasoning
- language-modeling architectures
- diffusion-specific reporting and rollout logic

## Near-term implementation guidance

1. Keep the new family small at first: metadata, package roots, and trainer
   integration seams.
2. Add explicit recursive reasoning contracts before adding benchmark-specific
   code.
3. Reuse existing trainer lifecycle, checkpoint, and report patterns from the
   byte-latent and diffusion OPD families instead of inventing a parallel
   infrastructure stack.

## Current reusable foundation

The first reusable surface is in
`gollek/training/gollek-train-recursive-reasoning` and is mapped as
`:ml:gollek-ml-recursive-reasoning`.

Use `./gradlew --no-daemon checkRecursiveReasoningTraining` from `gollek/` for
the focused verification gate, or `checkTrainingModules` for the aggregate
training-module gate.

Implemented surfaces:

- rollout/session contracts for sampled recursive trajectories
- stochastic latent transition interface
- reference width/depth rollout executor
- compact rollout report records
- benchmark-neutral token-pair dataset examples for trainer-facing structured
  reasoning batches, with N-Queens and Graph Coloring adapters
- compact token dataset profiling for task distribution, sequence-length stats,
  token range/distinct-token counts, and known-solution coverage diagnostics
- trainer-ready token dataset planner that composes profiling, split policy,
  train/evaluation epoch construction, padding config, and ordering policy
- compact dataset-plan diagnostics for pre-training health checks, including
  empty split warnings, dropped train examples, known-solution coverage, and
  high-padding detection, plus compact status labels, configurable warning
  policies, readiness gates, stable plan fingerprints, checkpoint/resume
  fingerprint guards, checkpoint metadata rehydration, one-stop preflight
  reports, typed checkpoint manifests, read-side manifest snapshots, and
  checkpoint-resume preflight reports with current-plan readiness checks,
  optional experiment identity expectations, rehydratable trainer resume
  policies, policy-tracked resume reports, deterministic checkpoint metadata
  JSON IO, trainer checkpoint-directory bridge helpers, lifecycle provenance
  listener support, read-side resume report snapshots, whole checkpoint
  directory snapshots, checkpoint inventory scans, reusable checkpoint
  selection policies for latest-ready versus latest-resume-ready restores,
  checkpoint inspection reports for JBang, dashboards, and CI restore gates,
  one-object trainer provenance specs, and metadata export for trainer reports,
  with per-task split coverage for mixed benchmark datasets
- backend-neutral padded token dataset batches with masks, lengths,
  known-solution counts, and per-example metadata for trainer handoff
- deterministic train/validation/test dataset splitter with exact-count and
  floor-fraction policies, task-stratified fraction splits for mixed benchmark
  datasets, plus convenience epoch builders for each partition
- deterministic token dataset epoch builder for sequential or seeded-shuffled
  mini-batches, length-sorted padding reduction, padding-efficiency metrics,
  partial-tail retention, and drop-last policies
- GRAM objective config and scalar breakdown records
- tensor-level diagonal Gaussian KL helper for prior/posterior heads
- weighted tensor objective for reconstruction, KL, LPRM, and ACT terms
- deterministic proposal, prior/posterior distribution-head, epsilon sampler,
  and next-state-factory contracts for reusable variational transitions
- shared Gaussian `float[]` noise helper in `core:gollek-tensor`, extracted from
  a Stable Diffusion latent-noise pattern so GRAM and diffusion-style code can
  converge on one seeded noise primitive
- reference deep-supervision trainer adapter that runs posterior transitions,
  evaluates terminal reconstruction/LPRM/ACT losses through a pluggable hook,
  and aggregates either paper-style truncated final-transition KL or full
  per-step KL for diagnostics
- first structured-reasoning benchmark adapter for N-Queens, including
  fixed-queen validation, conflict counting, paper-style board token encoding,
  and GRAM-style unique valid solution coverage reporting
- N-Queens solver/enumerator for exact known-solution counts, partial-board
  completions, and automatic coverage denominators
- deterministic N-Queens dataset example generator that masks complete solver
  solutions into partial-board inputs with reproducible seeds and known solution
  counts
- N-Queens token codec for predicted board outputs, including invalid-token,
  missing-row, and multi-queen-row diagnostics plus token-based coverage
- N-Queens rollout evaluator that scores final states from generic recursive
  trajectories via a pluggable state-to-token decoder, preserving selected
  trajectory metadata, per-sample validity, and coverage metrics
- backend-neutral N-Queens logit projectors and a GRAM next-state decorator so
  concrete model heads can attach predicted board tokens without coupling the
  benchmark evaluator to ONNX, safetensor, LiteRT, or GGUF internals
- shared discrete-token projection helpers for logits shaped `[item, vocab]`,
  extracted from N-Queens so Sudoku, ARC-style grids, graph labels, and future
  structured benchmarks can reuse the same validation and argmax behavior
- shared recursive-state token metadata helpers, extracted from N-Queens so
  future structured benchmarks can attach and decode predicted token grids
  without depending on N-Queens classes
- shared rollout token collector, extracted from N-Queens evaluation so
  structured benchmarks can reuse selected-aware final-state token reports
- shared discrete-token coverage aggregation for valid-rate, duplicate, and
  unique-solution accounting, with N-Queens now mapping board-specific validity
  into this generic report path
- shared discrete rollout evaluator that composes final-state token collection,
  benchmark-specific candidate scoring, selected trajectory metadata, and
  coverage reporting
- shared GRAM next-state decorator for attaching projected discrete tokens to
  recursive state metadata, with N-Queens and Graph Coloring wrappers using the
  same mechanism
- second structured benchmark adapter for Graph Coloring, proving the generic
  discrete rollout evaluator and coverage layer can support more than N-Queens
  while preserving task-specific validation and reports
- Graph Coloring exact solver/enumerator for known solution counts, partial
  fixed-color completions, and all-solution coverage denominators
- deterministic Graph Coloring dataset generator that masks exact solver
  completions into partial fixed-color train/eval examples with known solution
  counts
- backend-neutral Graph Coloring logit projector for raw node argmax,
  constrained legal-color projection, and fixed-color-preserving projection

Next implementation layer:

- learned prior and posterior transition modules backed by real model layers
- ACT halt-head and LPRM value-head adapters
- small benchmark extension that wires the generic rollout evaluator and token
  projection decorator to a concrete model adapter, then Sudoku or ARC-style
  grids
