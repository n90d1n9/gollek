# Gollek ML Recursive Reasoning

This module is the initial Java package root for recursive reasoning model
families in Gollek.

The first target is the GRAM paper:

- Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, and Sungjin
  Ahn. "Generative Recursive Reasoning." arXiv:2605.19376, 2026.
  DOI: `10.48550/arXiv.2605.19376`.

- GRAM introduces stochastic latent reasoning trajectories, variational
  training, and width-based inference scaling over parallel samples

Intended near-term scope:

- recursive-reasoning family metadata
- recursive reasoning rollout/session contracts
- latent-state and stochastic-transition interfaces
- in-memory rollout executor and compact reasoning report
- shared latent-state and trajectory contracts for recursive reasoning models
- GRAM-style stochastic transition interfaces
- GRAM-style variational objective primitives:
  - scalar objective breakdowns for reports, checkpoints, and dashboards
  - tensor-level diagonal Gaussian KL helper for prior/posterior heads
  - weighted reconstruction, KL, LPRM, and ACT loss composition
- GRAM-style variational transition primitives:
  - deterministic proposal contract
  - prior/posterior distribution heads
  - reparameterized epsilon sampling bridge
  - adapter from prior sampling to generic recursive rollouts
- GRAM-style deep-supervision training adapter:
  - posterior rollout over configured supervision steps
  - terminal reconstruction/LPRM/ACT loss evaluator hook
  - truncated final-transition KL aggregation by default
  - optional full step KL aggregation for diagnostics
- reusable structured-output helpers:
  - benchmark-neutral token-pair dataset example for trainer-facing
    structured reasoning batches
  - compact token dataset profiling for task mix, input/target length stats,
    observed token ranges, distinct-token counts, and known-solution coverage
  - trainer-ready token dataset planner that composes profiling,
    train/validation/test splitting, and epoch construction from one config
  - compact dataset-plan diagnostics for split health, known-solution
    coverage, dropped train examples, padding-efficiency warnings, compact
    status labels, configurable warning policies, readiness gates, per-task
    split coverage, stable plan fingerprints, checkpoint/resume fingerprint
    guards, checkpoint metadata rehydration, and one-stop preflight reports
    with typed checkpoint manifests, read-side manifest snapshots, and
    checkpoint-resume preflight reports with current-plan readiness checks,
    optional experiment identity expectations, rehydratable trainer resume
    policies, policy-tracked resume reports, deterministic checkpoint metadata
    JSON IO, trainer checkpoint-directory bridge helpers, lifecycle provenance
    listener support, read-side resume report snapshots, whole checkpoint
    directory snapshots, checkpoint inventory scans, reusable checkpoint
    selection policies for latest-ready versus latest-resume-ready restores,
    checkpoint inspection reports for JBang, dashboards, and CI restore gates,
    one-object trainer provenance specs, and report-friendly metadata export
  - backend-neutral padded token dataset batcher with input/target masks,
    sequence lengths, known-solution counts, and per-example metadata
  - deterministic train/validation/test token dataset splitter with exact-count
    and floor-fraction split policies, including task-stratified fraction
    splits for mixed benchmark datasets
  - deterministic mini-batch epoch builder with sequential/shuffled ordering,
    seeded shuffle, length-sorted padding reduction, padding-efficiency
    metrics, partial-tail retention, and drop-last support
  - backend-neutral discrete-token projection from `[item, vocab]` logits
  - shared argmax and best-item-for-token helpers for Sudoku, ARC-style grids,
    graph labels, and board benchmarks
  - generic recursive-state metadata helpers for attaching and decoding
    discrete token predictions independent of a benchmark family
  - generic rollout token collector for turning final states across sampled
    recursive trajectories into selected-aware token reports
  - generic discrete-token evaluation and coverage aggregation for valid-rate,
    duplicate, and unique-solution reporting
  - generic discrete rollout evaluator that composes token collection,
    benchmark-specific candidate scoring, selected trajectory tracking, and
    coverage aggregation
  - generic GRAM next-state decorator for attaching projected discrete tokens
    to recursive state metadata
- structured-reasoning benchmark adapters:
  - N-Queens problem/solution token encoding
  - fixed-queen, column, diagonal, and completion validation
  - multi-sample valid-rate and unique-solution coverage reporting
  - N-Queens completion solver for exact solution counts and coverage denominators
  - deterministic N-Queens dataset example generator for partial-board train/eval samples
  - N-Queens token decoder for model predictions with malformed-output diagnostics
  - rollout-to-N-Queens evaluation bridge for scoring recursive trajectories
    through pluggable state-to-token decoders
  - selected-trajectory, valid-rate, and unique-solution coverage reports for
    GRAM-style width-scaling experiments
  - backend-neutral N-Queens logit projection helpers for model heads:
    per-cell argmax, row-constrained queen argmax, and fixed-queen-preserving
    row projection
  - GRAM next-state decorator that attaches projected N-Queens tokens to
    recursive state metadata for later rollout evaluation
  - Graph Coloring problem/solution token encoding
  - fixed-color, edge-conflict, malformed-token, and completion validation
  - Graph Coloring exact solver/enumerator for known solution counts and
    coverage denominators
  - deterministic Graph Coloring dataset example generator for partial
    fixed-color train/eval samples
  - backend-neutral Graph Coloring logit projection helpers for model heads:
    raw node argmax, legal-color node argmax, and fixed-color-preserving
    legal-color projection
  - GRAM next-state decorator that attaches projected Graph Coloring tokens to
    recursive state metadata for later rollout evaluation
  - Graph Coloring rollout evaluator built on the generic discrete rollout
    evaluator and coverage layer
- trainer-facing config/session types for recursive reasoning experiments
- benchmark adapters for structured reasoning tasks such as Sudoku, ARC-style
  tasks, N-Queens, and Graph Coloring

Future family split:

- `ml:gollek-ml-reasoning-core`
- `ml:gollek-ml-recursive-reasoning`
- `trainer:gollek-trainer-recursive-reasoning`
- `examples:gollek-ml-examples` entries for GRAM-style probes and reports

See:

- `gollek/docs/GENERATIVE_RECURSIVE_REASONING_INTEGRATION_PLAN.md`
