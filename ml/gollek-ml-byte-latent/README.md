# Gollek ML Byte-Latent

This module is the initial Java package root for byte-latent language-modeling
work in Gollek.

It is intended for Fast Byte Latent Transformer style architectures and other
byte-native latent models that should remain separate from diffusion-specific
training modules.

Current scope:

- byte-sequence batch contract
- byte-level causal LM collator with padding and truncation windows
- lightweight UTF-8 text dataset bridge for byte batching
- trainer-facing config and planning helpers for byte-latent batches
- executable trainer session with builder and summary reporting
- byte-latent lifecycle listeners and lightweight checkpoint summary artifacts
- checkpoint resume from persisted byte-latent summary manifests
- persisted per-epoch history CSV for terminal-friendly training timelines
- canonical `byte-latent-report.json` with summary, history, and artifact paths
- compact history report selectors for `byte-latent-history.csv`
- byte-latent model spec and family metadata
- executable `ByteLatentModel` forward-pass contract with `ByteLatentForwardPass`
- deterministic `ReferenceByteLatentModel` for trainer-default model-backed loss
- prompt-to-next-token prediction and short byte-native generation defaults
- encoder/decoder interfaces
- latent-state transport object

Generation example:

```java
ByteLatentModel model = new ReferenceByteLatentModel(
        new ByteLatentModelSpec(256, 64, 2, 4, 16));
int nextToken = model.predictNextToken(new int[] {104, 105});
ByteLatentGenerationResult generated = model.generate(new int[] {104, 105}, 4);
```

History query examples:

- `history:summary`
- `history:lastLoss`
- `history:sort=-trainLoss:top=3`
- `history:epoch=2`

Checkpoint report examples:

- `overview`
- `summary`
- `summary:metadata:globalStep`
- `artifacts`

JBang workflow:

- `jbang trainer/trainer_byte_latent_demo.java`
- `jbang trainer/trainer_byte_latent_infer_demo.java`
- `jbang trainer/trainer_byte_latent_infer_demo.java "hi"`
- `jbang trainer/trainer_byte_latent_infer_demo.java "gollek" 6`
- `jbang trainer/trainer_byte_latent_train_infer_demo.java`
- `jbang trainer/trainer_byte_latent_train_infer_demo.java /tmp/byte-latent-train-infer "hi" 4`
- `jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer overview`
- `jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer ci`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo status`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo health`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo ci`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo status:ci`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo summary json`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_demo history:summary`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json overview`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json workflowBundleDir`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=health`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=files`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:focus=health:summary=short`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json status`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json health`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleStatus`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleHealth:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleHealth:summary=short`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:summary`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:ci:mode=fresh`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields:summary`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json inspectfields:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:all`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:summary`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runcommands:fresh`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:summary`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:ci`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json runfields:fresh`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:scope=workflow`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json commands:mode=fresh`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleOverview json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleManifest json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleFiles json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundleLoadfile:section=runs json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json delta:historyRows`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java trainer_checkpoints/byte_latent_workflow_manifest.json bundle=standard json /tmp/byte-latent-workflow-bundle`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleSummary json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleHealth json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:section=runs json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:section=runs:sort=name json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:name=overview.json json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle file:overview.json json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs:pick=last json`
- `jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs:index=0 json`

JBang resume workflow:

- `jbang trainer/trainer_byte_latent_resume_demo.java`
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_resume_demo overview json`

Inference walkthrough:

- `trainer_byte_latent_infer_demo.java`
- prints prompt token ids, `predictNextToken(...)`, generated token ids, and
  combined byte/text output using `ReferenceByteLatentModel`

Train+infer walkthrough:

- `trainer_byte_latent_train_infer_demo.java`
- runs a tiny trainer session, writes checkpoint artifacts, and then performs
  prompt continuation with the same byte-latent model family/spec
- also writes `train-infer-report.json` with machine-readable `training` and
  `inference` sections for CI or follow-up tooling
- `trainer_byte_latent_train_infer_inspector.java`
- reads `train-infer-report.json` from a checkpoint directory or direct file
- supports compact selectors such as `overview`, `training`, `inference`,
  `nextToken`, `generatedText`, `combinedText`, `status`, and `ci`
- `trainer_byte_latent_history_inspector.java` now also supports `status` as the
  shortest checkpoint-side status selector
- `ci` is the shortest compact CI-style checkpoint status alias
- `status:ci` remains available as the explicit CI-style checkpoint status form
- the demo and resume JBang scripts now print `status`, `health`, and `ci`
  follow-up commands automatically
- `jbang trainer/trainer_byte_latent_history_inspector.java trainer_checkpoints/byte_latent_resume_demo history:summary`

Smoke helper:

- `./trainer/run_byte_latent_workflow.sh`
- `./trainer/run_byte_latent_workflow.sh fresh`
- `./trainer/run_byte_latent_workflow.sh resume`
- `./trainer/run_byte_latent_workflow.sh full /tmp/gollek-byte-latent`
- the smoke helpers now lead with `status`, `health`, and `ci` inspector checks
- the workflow launcher now also prints `status`, `health`, and `ci` as its
  first checkpoint follow-up commands
- the workflow launcher now also prints `status` and `health` first for
  manifest and bundle follow-up commands
- the workflow launcher now also prints `commands:summary` for the manifest's
  compact command rollup
- the workflow launcher now also prints `runcommands:fresh` and
  `runcommands:resume` in that manifest follow-up block
- the workflow launcher now also prints `runcommands` there as the discovery
  selector for available per-run aliases
- the workflow launcher now also prints `runcommands:ci` there as the compact
  discovery summary for available per-run aliases
- the workflow launcher now also prints `runcommands:all` there as the concise
  multi-run alias summary
- the workflow launcher now also prints `inspectfields` there as the
  full machine-readable field-discovery selector
- the workflow launcher now also prints `inspectfields:ci` there as the
  compact machine-readable field-discovery summary
- the workflow launcher now also prints `runfields:ci` there as the compact
  per-run field-discovery summary
- the workflow launcher now also prints `runfields` there as the per-run
  field-discovery selector
- each fresh/resume run summary now also prints `runfields:ci` as the compact
  per-run field-discovery hint
- each fresh/resume run summary now also prints `runcommands:ci` as the
  generic compact discovery hint for available per-run aliases
- each fresh/resume run summary now also prints `runcommands:all` as the
  concise multi-run alias overview path
- each fresh/resume run summary now also prints `runcommands:<run>` alongside
  `commands:ci:mode=<run>` for the single-line manifest-driven CI follow-up
- `full` mode ends with a compact fresh/resume checkpoint summary
- the workflow launcher also writes a small `*-workflow-manifest.json`
- that manifest now records `workflowBundleDir` and `workflowBundleStatus`
- that manifest now also records `inspectWorkflowStatus`,
  `inspectWorkflowHealth`, `inspectBundleStatus`, and `inspectBundleHealth`
- each per-run manifest entry now also records `inspectWorkflowCommands` for
  the single-line `commands:ci:mode=<run>` workflow follow-up
- each per-run manifest entry now also records `inspectRunCommands` for the
  shortest `runcommands:<run>` workflow follow-up
- each per-run manifest entry now also records `inspectRunAll` for the
  concise multi-run `runcommands:all` workflow overview path
- each per-run manifest entry now also records `inspectRunFields` for the
  compact `inspectfields:ci` machine-readable field-discovery hint
- `bundleOverview` gives a compact terminal-friendly bundle summary from the
  manifest, with optional `focus=health|files|all`
- plain `text` output is optimized for quick terminal scanning
- `summary=short` collapses the text view to one line for CI logs
- `bundleOverview:ci` is the shorthand alias for the CI-style overview path
- `status` is the shortest neutral alias for a concise bundle check
- `health` is the matching shortest direct health alias for workflow and bundle checks
- `commands` exposes the manifest's recommended inspector commands as a table
  with optional `scope=` and `mode=` filtering
- `commands:summary` and `commands:ci` expose the compact top-priority command
  set for workflow, bundle, and fresh/resume checkpoints
- `commands:ci:mode=<run>` exposes the single-line per-run compact command rollup
- `runcommands` lists the available per-run aliases from the manifest
- `runcommands:all` expands to the concise multi-run alias summary
- `runcommands:summary` and `runcommands:ci` collapse that discovery view into
  compact manifest-level run-alias summaries
- `runcommands:<run>` is the shortest alias for that per-run manifest rollup
- `runfields` lists the per-run `inspectRunFields`, `inspectRunCommands`, and
  `inspectRunAll` follow-up aliases together
- `runfields:summary` and `runfields:ci` collapse that per-run field-discovery
  view into compact manifest-level summaries
- `runfields:<run>` isolates that per-run field-discovery row for one run mode
- `inspectfields` lists which machine-readable inspector-command fields are
  present at the manifest and per-run levels
- `inspectfields:summary` and `inspectfields:ci` collapse that field-discovery
  view into compact terminal and CI summaries
- `bundleStatus` remains available as the explicit equivalent
- `bundleHealth:ci` is the shorthand alias for that one-line direct health path
- in `full` mode it also materializes `byte-latent-workflow-bundle/`
- the workflow inspector can jump directly from that manifest into the
  materialized bundle with `bundleManifest`, `bundleFiles`, and
  `bundleLoadfile:section=...`
- `trainer_byte_latent_workflow_inspector.java` can query that manifest
- it can also export a small `bundle=standard` workflow artifact directory
- exported workflow bundles can be checked with `bundleSummary` and `bundleHealth`
- individual bundle entries can be queried with `file:` and `loadfile:section=...`
- bundle file lists can be filtered with `files:section=...` and `files:name=...`
- `files:` also supports `sort=name|section|format`
- `filesSummary` also supports `sort=-count` and `top=<n>`
- `loadfile:` also supports `pick=first|last`
- the launcher prints ready-to-run manifest and bundle inspector commands after
  each run
- `./trainer/run_byte_latent_smoke.sh`
- `./trainer/run_byte_latent_smoke.sh /tmp/gollek-byte-latent-smoke`

Resume smoke helper:

- `./trainer/run_byte_latent_resume_smoke.sh`
- `./trainer/run_byte_latent_resume_smoke.sh /tmp/gollek-byte-latent-resume-smoke`

Paper reference:

- Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz, Gargi Ghosh, Luke
  Zettlemoyer, Christopher Potts, Xiaochuang Han, and Srinivasan Iyer.
  "Fast Byte Latent Transformer." arXiv:2605.08044v1, 2026.

See also:

- `gollek/docs/FAST_BYTE_LATENT_TRANSFORMER_INTEGRATION_PLAN.md`
