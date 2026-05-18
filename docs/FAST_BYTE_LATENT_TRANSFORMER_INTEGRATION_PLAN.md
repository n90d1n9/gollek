# Fast Byte Latent Transformer Integration Plan

This note maps `papers/2605.08044v1.pdf` into Gollek's current codebase.

Paper citation:

- Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz, Gargi Ghosh, Luke
  Zettlemoyer, Christopher Potts, Xiaochuang Han, and Srinivasan Iyer.
  "Fast Byte Latent Transformer." arXiv:2605.08044v1, 2026.
  DOI: `10.48550/arXiv.2605.08044`.

## Architectural conclusion

Fast Byte Latent Transformer is a byte-level language-model architecture. It
should be treated as a new language-modeling feature family, not as an
enhancement to `training/gollek-train-diffusion-opd`.

That means:

- shared infra may be reused across Gollek core, tokenizer, runners, and
  trainer runtime
- architecture-specific encoder/decoder and latent-transformer components
  should live in new language/modeling modules
- diffusion-specific rollout and teacher-distillation code remains isolated in
  `training/gollek-train-diffusion-opd`

## Best-fit module direction

Recommended additions under the ML side:

- `ml:gollek-ml-language-core`
  - common language-modeling contracts, byte sequence batch shapes, and model
    metadata
- `ml:gollek-ml-byte-latent`
  - byte-latent encoder/decoder, latent transformer blocks, and latent cache
    abstractions
- `ml:gollek-ml-byte-io`
  - byte-native collation, packing, corruption/masking helpers, and dataset
    adapters when the existing token-oriented dataset helpers are not enough
- `runner:gollek-runner-byte-latent`
  - architecture-specific inference adapter for byte-latent checkpoints and
    runtime backends
- `trainer:gollek-trainer-byte-latent`
  - byte-native training/inference orchestration that reuses generic trainer
    lifecycle pieces without coupling to diffusion OPD

## Reuse points in the current repo

These are the code paths the new family should build on rather than duplicate:

- Core tensor/runtime primitives:
  - `gollek/core/gollek-core`
  - `gollek/core/gollek-tensor`
  - `gollek/core/gollek-ir`
- Token and dataset surfaces that may be extended toward bytes:
  - `gollek/core/gollek-tokenizer-core`
  - `gollek/training/gollek-train-data/src/main/java/tech/kayys/gollek/train/data/TokenizedDataset.java`
  - `gollek/training/gollek-train-data/src/main/java/tech/kayys/gollek/train/data/TextDataset.java`
- Training/runtime surfaces worth reusing:
  - `gollek/trainer/gollek-trainer/src/main/java/tech/kayys/gollek/trainer/CanonicalTrainerRuntime.java`
  - `gollek/training/gollek-train-api/src/main/java/tech/kayys/gollek/train/Gollek.java`
- Existing runner/backend seams:
  - `gollek/core/plugin/gollek-plugin-runner-core`
  - `gollek/runner/gguf/gollek-gguf-core`
  - `gollek/runner/safetensor`
  - `gollek/backend/metal`
  - `gollek/backend/rocm`

## What should not happen

Do not land Fast Byte Latent Transformer as:

- a feature inside `training/gollek-train-diffusion-opd`
- a small extension of diffusion teacher/student abstractions
- a tokenizer-only enhancement without dedicated latent-model modules

Those paths would blur the boundary between diffusion-specific training logic
and byte-level language modeling, and would make future maintenance harder.

## Near-term implementation guidance

1. Keep the current DiffusionOPD work focused on diffusion reports, adapters,
   and rollout logic.
2. Introduce byte-latent module stubs and package roots as a new family rather
   than threading them through diffusion packages.
3. Reuse trainer lifecycle, report/export, and runner registration seams from
   the code paths cited above.
