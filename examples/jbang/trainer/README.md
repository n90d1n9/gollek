# Trainer JBang Examples

This folder is the canonical JBang lane for Gollek trainer runtime examples.

## Run

From `gollek`, refresh the Gradle-built local artifacts first:

```bash
./gradlew publishJbangTrainerExamplesToMavenLocal
./gradlew smokeJbangTrainerRuntimeProfileBudgetGate
./gradlew verifyJbangTrainerRuntimeProfileBudgetGateOutput
./gradlew smokeJbangTrainerQualityProfileCiGateEvidence
./gradlew verifyJbangTrainerQualityProfileCiGateEvidenceOutput
./gradlew verifyJbangTrainerEvidenceIndex
./gradlew smokeJbangTrainerExamples
```

The Gradle smoke lane runs JBang with an isolated `build/jbang/.jbang` home and
offline dependency resolution after publishing the local Gradle artifacts. Use
`-Pgollek.jbang.trainer.offline=false` only when intentionally refreshing remote
JBang dependencies.

From `gollek/examples/jbang`:

```bash
jbang trainer/trainer_runtime_bootstrap.java
jbang trainer/trainer_runtime_profile_budget_gate.java
jbang trainer/trainer_runtime_profile_budget_gate.java /tmp/gollek-runtime-budget --policy strict
jbang trainer/trainer_quality_profile_ci_gate_evidence.java
jbang trainer/trainer_quality_profile_ci_gate_evidence.java /tmp/gollek-ci-evidence --device auto --profile local-experiment
jbang trainer/trainer_diffusion_opd_ddim.java
jbang trainer/trainer_diffusion_opd_ddim.java 4
jbang trainer/trainer_diffusion_opd_stable_diffusion_metal_bridge.java
jbang trainer/trainer_diffusion_opd_stable_diffusion_native_bridge.java
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json overview
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json conditioning
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json teacherSummaries json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 csv
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table value,count,meanLoss
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 csv pair,count,meanLoss
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table value,count,loss,latestRound,latestLoss
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table compact
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 csv latest
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table details
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 table minimal
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 table compact
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 csv compare
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummaries:sort=-meanLoss:top=5 csv compact /tmp/opd-task-summaries.csv
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 table compare /tmp/opd-task-teacher-compare.txt
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummary=ocr json /tmp/opd-task-summary-ocr.json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummary=ocr /tmp/opd-task-summary-ocr.json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 /tmp/opd-task-teacher-summaries.csv
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=standard json /tmp/opd-standard-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=rollups csv /tmp/opd-rollups-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=custom:overview,taskSummaries,taskTeacherSummaries csv /tmp/opd-custom-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=custom:overview@run-overview,taskTeacherSummaries@teacher-pairs csv /tmp/opd-custom-aliased-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=custom:taskSummaries@tasks#compact,taskTeacherSummaries@pairs#compare csv /tmp/opd-custom-columned-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=custom:overview@run-overview!json,taskTeacherSummaries@teacher-pairs#compare!csv /tmp/opd-custom-mixed-format-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json bundle=standard /tmp/opd-standard-text-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleSummary json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleHealth json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleHealth:focus=sections json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleHealth:focus=files:top=5 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleSummary:top=3 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleSummary:focus=sections:top=3 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle bundleSummary:dominant=true json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle files json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle files:format=csv json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle files:section=taskTeacherSummaries json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle filesSummary:by=section json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle filesSummary:by=format:sort=-count json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle filesSummary:by=section:sort=-count:top=3 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle files:section=taskTeacherSummaries:sort=name json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle file:overview.json json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle loadfile:overview.json
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle loadfile:format=csv:section=taskTeacherSummaries
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle loadfile:format=csv:section=taskTeacherSummaries:pick=last
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle loadfile:format=csv:section=taskTeacherSummaries:sort=-name:pick=first
jbang trainer/trainer_diffusion_opd_report_inspector.java /tmp/opd-standard-bundle loadfile:format=csv:index=1
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummaries:sort=-meanLoss:top=10 json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskSummary=ocr json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json teacherSummary=ocr-early json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskTeacherSummary=ocr,ocr-early json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json taskStageSummary=ocr,early json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory:last json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory:task=ocr json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory:task=ocr:last json
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory:task=ocr:meanLoss
jbang trainer/trainer_diffusion_opd_report_inspector.java /path/to/diffusion-opd-report.json roundHistory:task=ocr:sort=-averageLoss:top=3 json
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/byte-latent-history.csv
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir status
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir health
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir ci
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir status:ci
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir history:summary
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir history:lastLoss
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir history:sort=-trainLoss:top=3 json
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir summary json
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir summary:metadata:globalStep
jbang trainer/trainer_byte_latent_history_inspector.java /path/to/checkpoint-dir history:epoch=2 csv /tmp/byte-latent-epoch-2.csv
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json overview
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runs json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json workflowBundleDir
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json workflowBundleStatus
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview:focus=health
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview:focus=files
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview:focus=health:summary=short
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json status
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json health
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleStatus
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleHealth:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleHealth:summary=short
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands:summary
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands:ci:mode=fresh
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json inspectfields
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json inspectfields:summary
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json inspectfields:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runcommands
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runcommands:all
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runcommands:summary
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runcommands:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runcommands:fresh
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runfields
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runfields:summary
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runfields:ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json runfields:fresh
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands:scope=workflow
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json commands:mode=fresh
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleOverview json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleManifest json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleFiles json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleSummary json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundleLoadfile:section=runs json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json run:fresh json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json delta:historyRows
jbang trainer/trainer_byte_latent_infer_demo.java
jbang trainer/trainer_byte_latent_infer_demo.java "hi"
jbang trainer/trainer_byte_latent_infer_demo.java "gollek" 6
jbang trainer/trainer_byte_latent_train_infer_demo.java
jbang trainer/trainer_byte_latent_train_infer_demo.java /tmp/byte-latent-train-infer "hi" 4
jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer overview
jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer training
jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer nextToken
jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer combinedText
jbang trainer/trainer_byte_latent_train_infer_inspector.java /tmp/byte-latent-train-infer ci
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json delta:latestTrainLoss json
jbang trainer/trainer_byte_latent_workflow_inspector.java /path/to/byte_latent_workflow_manifest.json bundle=standard json /tmp/byte-latent-workflow-bundle
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleSummary json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle bundleHealth json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:section=runs json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:section=runs:sort=name json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle files:name=overview.json json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle filesSummary:by=section:sort=-count:top=3 json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle file:overview.json json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs json
jbang trainer/trainer_byte_latent_workflow_inspector.java /tmp/byte-latent-workflow-bundle loadfile:section=runs:index=0 json
jbang trainer/trainer_byte_latent_demo.java
jbang trainer/trainer_byte_latent_demo.java /tmp/gollek-byte-latent-demo
jbang trainer/trainer_byte_latent_resume_demo.java
jbang trainer/trainer_byte_latent_resume_demo.java /tmp/gollek-byte-latent-resume-demo
./trainer/run_byte_latent_workflow.sh
./trainer/run_byte_latent_workflow.sh fresh
./trainer/run_byte_latent_workflow.sh resume
./trainer/run_byte_latent_workflow.sh full /tmp/gollek-byte-latent
./trainer/run_byte_latent_smoke.sh
./trainer/run_byte_latent_smoke.sh /tmp/gollek-byte-latent-smoke
./trainer/run_byte_latent_resume_smoke.sh
./trainer/run_byte_latent_resume_smoke.sh /tmp/gollek-byte-latent-resume-smoke
jbang trainer/trainer_accelerated_autograd.java auto
jbang trainer/trainer_accelerated_autograd.java metal
jbang trainer/trainer_warmup_cosine_scheduler.java auto
jbang trainer/trainer_warmup_cosine_scheduler.java metal
jbang trainer/trainer_reduce_lr_on_plateau.java auto
jbang trainer/trainer_reduce_lr_on_plateau.java metal
jbang trainer/trainer_resume_history.java auto
jbang trainer/trainer_resume_history.java metal
jbang trainer/trainer_robust_regression_huber.java auto
jbang trainer/trainer_robust_regression_huber.java metal
jbang trainer/trainer_classification_metrics.java auto
jbang trainer/trainer_classification_metrics.java metal
jbang trainer/trainer_imbalanced_crossentropy_weighted.java auto
jbang trainer/trainer_imbalanced_crossentropy_weighted.java metal
jbang trainer/trainer_focal_classification.java auto
jbang trainer/trainer_focal_classification.java metal
jbang trainer/trainer_binary_focal_weighted.java auto
jbang trainer/trainer_binary_focal_weighted.java metal
jbang trainer/trainer_binary_bce_metrics.java auto
jbang trainer/trainer_binary_bce_metrics.java metal
jbang trainer/trainer_imbalanced_bce_weighted.java auto
jbang trainer/trainer_imbalanced_bce_weighted.java metal
jbang trainer/trainer_multilabel_bce_metrics.java auto
jbang trainer/trainer_multilabel_bce_metrics.java metal
```

## What This Example Covers

- Trainer runtime mode detection (`legacy-bridge` vs `canonical-fallback`)
- Java-first DiffusionOPD demo with `Gollek.DL.diffusionOpdTrainer()`
- Existing diffusion runner adapter usage via `RunnerDiffusionAdapters`
- Stable-diffusion-native Metal bridge usage via
  `StableDiffusionRunnerAdapters.metalDefaultTensorFloat32Bridge(...)`
- End-to-end stable-diffusion-native OPD path using both
  `StableDiffusionRunnerAdapters.denoiser(...)` and executable
  `StableDiffusionRunnerAdapters.scheduler(...)`
- DDIM-style ODE mean-matching rollout scaffold for diffusion distillation
- Canonical trainer builder usage (`Trainers.canonicalBuilder()`)
- Training listener lifecycle callbacks
- Training summary reporting
- Runtime profile budget gate lifecycle with JSON/Markdown/JUnit artifact
  writing and verification
- One-command byte-latent smoke flow for fresh runs and resumed runs
- Unified byte-latent workflow runner with `fresh`, `resume`, and `full` modes
- Compact end-of-run workflow summary with fresh/resume checkpoint details
- Machine-readable workflow manifest JSON for CI or follow-up scripts
- Persisted `workflowBundleDir` and `workflowBundleStatus` fields in that
  manifest
- Persisted `inspectWorkflowStatus`, `inspectWorkflowHealth`,
  `inspectBundleStatus`, and `inspectBundleHealth` fields in that manifest
- Persisted per-run `inspectWorkflowCommands` fields in that manifest for the
  single-line `commands:ci:mode=<run>` workflow follow-up
- Persisted per-run `inspectRunCommands` fields in that manifest for the
  shortest `runcommands:<run>` workflow follow-up
- Persisted per-run `inspectRunAll` fields in that manifest for the concise
  multi-run `runcommands:all` workflow overview path
- Persisted per-run `inspectRunFields` fields in that manifest for the compact
  `inspectfields:ci` machine-readable field-discovery hint
- Compact `bundleOverview` manifest selector with `focus=health|files|all`
- Text output for `bundleOverview` is optimized for terminal scanning
- `bundleOverview:...:summary=short` emits a single-line CI-friendly summary
- `bundleOverview:ci` is the shorthand alias for that CI-friendly overview
- `status` is the shortest neutral alias when you just want the concise
  bundle check
- `health` is the matching shortest direct health alias on the workflow and
  bundle path
- `commands` renders the manifest's recommended inspector commands as a small
  table, with optional filters like `scope=workflow` or `mode=fresh`
- `commands:summary` and `commands:ci` collapse that to the most important
  workflow, bundle, and fresh/resume checkpoint status-health commands
- `commands:ci:mode=<run>` emits the single-line per-run compact command rollup
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
- `bundleStatus` remains available as the more explicit equivalent
- `bundleHealth:ci` is a shorthand alias for the single-line direct health view
- `bundleHealth:summary=short` emits the same single-line direct health summary
- Matching `status` shorthand on the checkpoint/history inspector
- Matching `health` shorthand on the checkpoint/history inspector
- `ci` is the shortest checkpoint-side CI status alias
- `status:ci` on the checkpoint/history inspector remains the explicit CI form
- The workflow launcher now prints `status`, `health`, and `ci` checkpoint
  follow-ups before the longer inspector examples
- The workflow launcher now also prints `status` and `health` first for
  workflow manifest and bundle follow-up checks
- The workflow launcher now also prints `commands:summary` first for the
  workflow-manifest compact command rollup
- Each fresh/resume run summary now also prints `runcommands:ci` as the
  generic compact discovery hint for available per-run aliases
- Each fresh/resume run summary now also prints `runcommands:all` as the
  concise multi-run alias overview path
- The workflow launcher now also prints `runcommands:fresh` and
  `runcommands:resume` in that workflow-manifest block
- The workflow launcher now also prints `runcommands` there as the discovery
  selector for available per-run aliases
- The workflow launcher now also prints `runcommands:ci` there as the compact
  discovery summary for available per-run aliases
- The workflow launcher now also prints `runcommands:all` there as the concise
  multi-run alias summary
- The workflow launcher now also prints `inspectfields` there as the
  full machine-readable field-discovery selector
- The workflow launcher now also prints `inspectfields:ci` there as the
  compact machine-readable field-discovery summary
- The workflow launcher now also prints `runfields:ci` there as the compact
  per-run field-discovery summary
- The workflow launcher now also prints `runfields` there as the per-run
  field-discovery selector
- Each fresh/resume run summary now also prints `runfields:ci` as the compact
  per-run field-discovery hint
- Each fresh/resume run summary now also prints
  `runcommands:<run>` alongside `commands:ci:mode=<run>` for the shortest
  manifest-driven CI follow-up
- Automatic workflow-bundle materialization in `full` mode for immediate
  `loadfile:` inspection
- Direct manifest-to-bundle selectors for materialized bundle inspection
- Workflow manifest inspector for `runs`, per-mode views, and fresh/resume deltas
- End-of-run workflow-inspector commands printed directly by the launcher
- End-of-run workflow-bundle export and `loadfile:` examples printed directly
  by the launcher
- Tiny byte-latent inference walkthrough with `predictNextToken(...)` and
  short deterministic `generate(...)` continuation output
- Tiny combined train-then-infer walkthrough for one-command lifecycle
  verification, including a machine-readable `train-infer-report.json`
- Matching `trainer_byte_latent_train_infer_inspector.java` for compact
  `overview`, `training`, `inference`, `nextToken`, `generatedText`,
  `combinedText`, `status`, and `ci` queries over that report
- Workflow bundle export for shareable overview/runs/delta artifact directories
- Workflow bundle summary/health inspection for exported artifact directories
- Direct bundle file lookup and section-based bundle file loading
- `files:` sorting by `name`, `section`, or `format`
- Compact filtered bundle file listing by section or filename
- Public `Gollek.DL.fit(...)` training options
- Public `Gollek.DL.trainValidationSplit(...)` plus deterministic seeded
  train loaders
- Stratified `Gollek.DL.classificationStratifiedTrainValidationSplit(...)`
  `.binaryStratifiedTrainValidationSplit(...)`, and
  `.multiLabelBinaryStratifiedTrainValidationSplit(...)` for label-safe
  classification validation sets
- Direct `Gollek.DL.fit(model, split, ...)` split training
- `Gollek.DL.trainingOptions().meanAbsoluteErrorMetric().mseMetric()` for
  train/validation metrics in the returned training summary
- `Gollek.DL.trainingOptions().regressionMetrics()` for MAE, MSE, RMSE, and
  R2 reporting on regression loaders
- `Gollek.DL.trainingOptions().checkpointDir(...).restoreBestModelAtEnd()` for
  validation-driven best-model snapshots in `canonical-best-model.safetensors`;
  add `.bestModelMonitorMetric("f1", CanonicalTrainer.BestModelMonitorMode.MAX)`
  when a task metric should choose the best epoch instead of validation loss
- `Gollek.DL.trainingOptions().earlyStopping(...).earlyStoppingMonitorMetric(...)`
  when early stopping should follow a validation metric such as F1, accuracy,
  AUROC, or average precision instead of validation loss
- `TrainingSummary.metadata().get("epochHistory")` for per-epoch train loss,
  validation loss, metric values, learning rate, optimizer steps, and scheduler
  steps so trainer runs can be debugged after completion
- Gradient and parameter diagnostics in summary/history metadata, including
  `latestGradientL2NormBeforeClip`, `latestGradientL2Norm`,
  `latestGradientClipped`, and `latestParameterL2Norm`
- Non-finite guard metadata (`nonFiniteDetected`, `nonFinitePhase`,
  `nonFiniteKind`, `nonFiniteOptimizerStepSkipped`) when NaN/Inf loss or
  gradients stop training before corrupted checkpoints are written
- Train/validation throughput diagnostics in summary/history metadata,
  including `trainBatchCount`, `trainSampleCount`, `trainSamplesPerSecond`,
  `validationBatchCount`, and `validationSamplesPerSecond`
- `canonical-history.csv` next to `checkpointDir(...)` checkpoints for a durable
  learning-curve artifact that can be opened in spreadsheets or dashboards;
  `resumeFromCheckpoint()` reloads existing rows before appending new epochs
- `canonical-report.json` next to `checkpointDir(...)` checkpoints for a
  structured run report containing summary metadata, epoch history, accelerator
  selection, checkpoint state, gradient diagnostics, and throughput counters
- Optimizer checkpoint state for SGD, Adam, AdamW, and RMSprop so interrupted
  runs resume with momentum/velocity buffers and Adam moments intact
- `Gollek.DL.TrainingPreset.REGRESSION_HUBER_*` and
  `Gollek.DL.huberLoss(...)` for robust regression with outliers
- `Gollek.DL.trainingOptions().classificationMetrics()` for accuracy, macro
  precision, macro recall, and macro F1 on CrossEntropy-style classification
  loaders
- `Gollek.DL.confusionMatrixMetric()` and
  `.trainingOptions().confusionMatrixMetric()` for structured multi-class
  confusion matrix details in summary metadata, epoch history, and
  `canonical-report.json`
- `Gollek.DL.trainingOptions().topKAccuracyMetric(k)` for top-k
  classification reporting
- `Gollek.DL.classificationMacroRocAucMetric()`,
  `.classificationMacroAveragePrecisionMetric()`, and
  `.trainingOptions().classificationRankingMetrics()` for one-vs-rest
  multi-class ranking quality from CrossEntropy-style logits
- `Gollek.DL.classWeights(...)` / `.classWeightsFor(numClasses, ...)` and
  `.trainingOptions().crossEntropyClassWeights(...)` for imbalanced
  multi-class CrossEntropy training
- `Gollek.DL.TrainingPreset.CLASSIFICATION_FOCAL_*`,
  `Gollek.DL.focalLoss(...)`, and `.trainingOptions().focalGamma(...)` /
  `.focalClassWeights(...)` for hard-example-focused classification training
- `Gollek.DL.TrainingPreset.BINARY_FOCAL_WITH_LOGITS_*`,
  `Gollek.DL.binaryFocalWithLogitsLoss(...)`, and
  `.trainingOptions().focalGamma(...)` / `.focalAlpha(...)` with
  `.bcePositiveWeight(...)` for imbalanced binary or multi-label focal
  training
- `Gollek.DL.binaryStratifiedTrainValidationSplit(...)` and
  `Gollek.DL.trainingOptions().binaryClassificationMetrics()` for
  BCE-with-logits binary classification
- `Gollek.DL.binaryAccuracyMetric(logitThreshold)` and
  `.trainingOptions().binaryClassificationMetrics(logitThreshold)` for
  thresholded binary metrics at deployment-specific logit operating points
- `Gollek.DL.binaryConfusionMatrixMetric(logitThreshold)` and
  `.trainingOptions().binaryConfusionMatrixMetric(logitThreshold)` for
  structured TN/FP/FN/TP diagnostics, specificity, balanced accuracy, and a
  `[[TN, FP], [FN, TP]]` matrix in trainer summaries and reports
- `Gollek.DL.binaryRocAucMetric()`, `.binaryAveragePrecisionMetric()`, and
  `.trainingOptions().binaryRankingMetrics()` for imbalanced binary ranking
  quality beyond a fixed threshold
- `Gollek.DL.TrainingPreset.BINARY_BCE_WITH_LOGITS_ADAMW` for one-output
  binary classifiers shaped `[batch, 1]`
- `Gollek.DL.binaryPositiveWeight(...)` and
  `.trainingOptions().bcePositiveWeight(...)` for imbalanced binary BCE
  training without hand-wiring a custom loss
- `Gollek.DL.multiLabelBinaryStratifiedTrainValidationSplit(...)` for
  multi-label BCE targets shaped `[batch, labels]`
- `Gollek.DL.multiLabelPositiveWeights(...)` plus
  `.trainingOptions().bcePositiveWeights(...)` for per-label imbalanced
  multi-label BCE training
- `Gollek.DL.trainingOptions().multiLabelBinaryMetrics()` for exact-match,
  Hamming loss, macro precision, macro recall, and macro F1 reporting on
  multi-label BCE outputs
- `Gollek.DL.multiLabelMacroF1Metric(logitThreshold)` and
  `.trainingOptions().multiLabelBinaryMetrics(logitThreshold)` for
  thresholded multi-label metrics at a chosen logit operating point
- `Gollek.DL.multiLabelMacroRocAucMetric()`,
  `.multiLabelMacroAveragePrecisionMetric()`, and
  `.trainingOptions().multiLabelRankingMetrics()` for per-label macro
  ranking quality on multi-label BCE/focal outputs
- `Gollek.DL.trainingOptions().stepLrBatches(...)` and
  `.cosineAnnealingLrBatches(...)` for public LR scheduling
- `Gollek.DL.trainingOptions().warmupCosineLrBatches(...)` and
  `.warmupCosineLrEpochs(...)` for Transformer-style linear warmup followed
  by cosine decay, with scheduler state included in summary metadata and
  `canonical-report.json`
- `Gollek.DL.trainingOptions().reduceLrOnPlateauValidationLoss(...)` and
  `.reduceLrOnPlateauMetric(...)` for validation-driven learning-rate
  reductions with checkpointable scheduler state and report metadata
- Public `Gollek.DL.evaluate(...)` for no-grad validation/test loss and
  metrics
- End-to-end toy classification with `trainer_classification_metrics.java`
- End-to-end quality-profile CI evidence with
  `trainer_quality_profile_ci_gate_evidence.java`
  - Trains tiny baseline and candidate classifiers into canonical report files
  - Runs `local-experiment` by default through the quality-profile CI gate
  - Writes the CI manifest, JSON/Markdown/JUnit manifest verification report,
    and verifies the persisted evidence bundle
  - Supports `--out`, `--device`, `--profile`, `--baseline-epochs`,
    `--candidate-epochs`, and `--fail-on-gate`
- End-to-end interrupted/resumed regression with durable history in
  `trainer_resume_history.java`
- End-to-end robust regression with `trainer_robust_regression_huber.java`
- End-to-end warmup/cosine scheduling with
  `trainer_warmup_cosine_scheduler.java`
- End-to-end validation plateau scheduling with
  `trainer_reduce_lr_on_plateau.java`
- End-to-end imbalanced multi-class classification with
  `trainer_imbalanced_crossentropy_weighted.java`
- End-to-end focal-loss classification with `trainer_focal_classification.java`
- End-to-end imbalanced binary focal-loss classification with
  `trainer_binary_focal_weighted.java`
- End-to-end binary classification with `trainer_binary_bce_metrics.java`
- End-to-end imbalanced binary classification with
  `trainer_imbalanced_bce_weighted.java`
- End-to-end multi-label classification with
  `trainer_multilabel_bce_metrics.java`
- Honest accelerator metadata (`executionBackend`, `executionAccelerated`,
  `acceleratedMatmulCalls`)
- End-to-end Java/JBang diffusion distillation walkthrough with
  `trainer_diffusion_opd_ddim.java`
- End-to-end Java/JBang Stable Diffusion Metal bridge walkthrough with
  `trainer_diffusion_opd_stable_diffusion_metal_bridge.java`
- End-to-end Java/JBang stable-diffusion-native OPD trainer walkthrough with
  `trainer_diffusion_opd_stable_diffusion_native_bridge.java`
  - Accepts an optional model root path argument and also checks
    `GOLLEK_SD_NATIVE_MODEL_DIR`
  - Validates the native runner layout: `text_encoder/`, `unet/`, `vae/`,
    and `tokenizer/`
  - Auto-upgrades from stub UNet to a real local safetensor-native UNet when
    the fixture is present
  - Auto-upgrades prompt conditioning from a synthetic tensor to real CLIP
    embeddings loaded from `text_encoder/` plus `tokenizer/` when the shared
    fixture is usable
  - Supports task-specific text-conditioning overrides such as
    `GOLLEK_SD_NATIVE_MODEL_DIR_TEXT_OCR` and
    `GOLLEK_SD_NATIVE_MODEL_DIR_TEXT_AESTHETICS`, with `text-ocr/` and
    `text-aesthetics/` subdirectory fallback under a shared fixture
  - Writes `conditioningTaskModes`, `conditioningTaskFixtureBaseDirs`, and
    `conditioningTaskFixtures` into the OPD summary/history artifacts so
    per-task text-conditioning lanes stay visible during loss analysis
  - Also records observed `conditioningTaskResolveCounts`,
    `conditioningTaskLossSteps`, `conditioningTaskLossSums`, and
    `conditioningTaskMeanLoss` so artifact analysis can distinguish configured
    lanes from lanes actually exercised during the run
  - Uses the formal `DiffusionOpdRuntimeObserver` path instead of example-only
    mutable metadata injection, so the same diagnostics pattern can be reused
    by other diffusion integrations
  - Uses the built-in `ConditioningTaskMetricsObserver` from
    `gollek-ml-diffusion-opd`, so per-task conditioning diagnostics no longer
    need custom observer classes in each example
  - Also uses the built-in `TeacherStageTaskMetricsObserver` for observed
    task, teacher, stage, and task-stage loss/usage aggregation during OPD
    rollout
  - Can now enable the standard OPD diagnostics bundle in one call through
    `DiffusionOpdDiagnosticsPacks.standardTaskDiagnostics(...)` together with
    `.runtimeObservers(...)`
  - Writes a normalized `diffusion-opd-report.json` artifact alongside
    `diffusion-opd-summary.json` and `diffusion-opd-history.csv` for easier
    post-run analysis of task, teacher, stage, conditioning, and adaptive
    diagnostics
  - The normalized report now follows typed schema records including
    `DiffusionOpdReport`, `DiffusionOpdRunReport`, and
    `DiffusionOpdArtifactsReport`
  - `DiffusionOpdReports.load(...)` can read `diffusion-opd-report.json` back
    into the typed schema for downstream Java tooling
- DiffusionOPD artifact inspection with
  `trainer_diffusion_opd_report_inspector.java`
  - Loads `diffusion-opd-report.json` through `DiffusionOpdReports.load(...)`
  - Supports `all`, `overview`, `run`, `artifacts`, `teachers`, `stages`,
    `tasks`, `conditioning`, `adaptive`, `bindings`, `roundHistory`, and
    `roundHistoryCount` section modes
  - Supports `text`, pretty-printed `json`, terminal-friendly `table`, and
    spreadsheet-ready `csv` output modes
  - Supports optional file export for `json`, `table`, `csv`, and `text`
    through a fifth argument output path
  - Can infer export format directly from output filenames like `.json`,
    `.csv`, `.table`, `.md`, `.txt`, or `.log` when the explicit format
    argument is omitted
  - Supports bundle export through selectors like `bundle=standard` and
    `bundle=rollups`, writing multiple related views into one output directory
  - Supports ad hoc bundle export with `bundle=custom:section1,section2,...`
    for small one-off analysis packs
  - Custom bundles also support inline output aliases through
    `section@alias`, for example
    `bundle=custom:overview@run-overview,taskTeacherSummaries@teacher-pairs`
  - Custom bundles support per-entry columns through `#preset` or
    `#value,count,loss`, for example
    `bundle=custom:taskSummaries@tasks#compact,taskTeacherSummaries@pairs#compare`
  - Custom bundles support per-entry formats through `!json`, `!csv`,
    `!table`, or `!text`, for example
    `bundle=custom:overview@run-overview!json,taskTeacherSummaries@teacher-pairs#compare!csv`
  - Bundle exports also write a `manifest.json` with bundle type, output
    format, creation timestamp, source report path, and per-file section
    metadata
  - Bundle directories and `manifest.json` files can be inspected directly,
    including `manifest`, `bundleSummary`, `bundleHealth`, `files`,
    `filesSummary:by=section`,
    `filesSummary:by=format:sort=-count`,
    `filesSummary:by=section:sort=-count:top=3`, `files:format=csv`,
    `files:section=taskTeacherSummaries`, `file:<name>`, and
    `loadfile:<name>` / `loadfile:format=csv:section=taskTeacherSummaries`
    views, with `sort=name|-name|section|format|count`, `top=<n>`,
    `pick=first|last`, or `index=<n>` when a filter matches more than one
    file
  - `bundleSummary` also supports `focus=sections|formats|all` plus `top=<n>`
    for compact embedded rollups, and `dominant=true` for headline-only output
  - `bundleSummary` also validates manifest entries against the bundle
    directory and reports `existingFileCount`, `missingFileCount`, and
    `missingFiles` when exported files are missing on disk
  - `bundleHealth` provides a compact integrity view with `healthy`,
    `status`, `healthScore`, `alertLevel`, `issueCodes`, `primaryIssueCode`,
    `healthBadge` (`status`, `alertLevel`, `primaryIssueCode`, `score`,
    `primaryCheckCode`, `primaryCheckSeverity`, `variant`, `label`, `token`, `tooltip`,
    `checkStatus`),
    `summaryMessage`, `recommendedAction`, `missingFileCount`, `missingFiles`,
    `missingSections`, `missingFormats`, structured `checks`, normalized
    `checkSummary` (`total`, `passed`, `failed`, `passRate`, `failureRate`,
    `status`, `alertLevel`, local `summaryMessage`, local `recommendedAction`,
    local `issueCodes`, local `primaryIssueCode`,
    nested `healthBadge` (`status`, `alertLevel`, `primaryIssueCode`,
    `primaryCheckName`, `primaryCheckMessage`, `primaryCheckSeverity`, `score`, `variant`, `label`, `token`,
    `checkStatus`, `summaryMessage`, `recommendedAction`, `tooltip`), `dominantSeverity`,
    `allPassed`, `hasCriticalFailures`, `hasWarningFailures`,
    `hasInfoFailures`, `failingSeverityCounts`, `primaryFailingName`, `primaryFailingCode`,
    `primaryFailingSeverity`, `primaryFailingMessage`,
    `failingNames`, `failingCodes`, `severityCounts`), plus
    `failingChecks`, `failingCheckCount`, `passingCheckCount`,
    `criticalCheckCount`, `warningCheckCount`, and `infoCheckCount`
  - `bundleHealth` also supports `focus=files|sections|formats|all` and
    `top=<n>` for trimmed diagnosis views
  - For `table` and `csv` bundles, grouped summaries default to `compact`
    columns and pair rollups default to `compare` columns unless an explicit
    projection is provided
  - Supports optional column projection for `table` and `csv`, for example
    `value,count,meanLoss` or `pair,count,meanLoss`
  - Supports friendly projection aliases such as `loss`, `latestRound`,
    `latestLoss`, `latestTeacher`, `latestTask`, `latestStage`, `top1Round`,
    `top1Loss`, `first1Round`, and `first1Loss`
  - Supports projection presets such as `minimal`, `compact`, `latest`,
    `leaderboard`, `details`, and `compare`
  - The richer `details` and pair-oriented `compare` presets now include both
    top-loss and first-round representative columns through `top1*` and
    `first1*` aliases
  - Presets are section-aware, so pair rollups automatically use `pair` while
    grouped single-dimension rollups use `value`
  - Supports grouped rollups such as `taskSummaries`, `teacherSummaries`, and
    `stageSummaries`
  - Supports ranked grouped rollups such as
    `taskSummaries:sort=-meanLoss:top=5` and
    `teacherSummaries:sort=-count:top=3`
  - Supports grouped pair rollups such as `taskTeacherSummaries`,
    `taskStageSummaries`, and `teacherStageSummaries`, including ranked forms
    like `taskTeacherSummaries:sort=-meanLoss:top=10`
  - Supports compact summary presets such as `taskSummary=ocr`,
    `teacherSummary=ocr-early`, and `stageSummary=early`
  - Supports paired summary presets such as
    `taskTeacherSummary=ocr,ocr-early`,
    `taskStageSummary=ocr,early`, and
    `teacherStageSummary=ocr-early,early`
  - Supports focused round-history selectors such as `roundHistory:last`,
    `roundHistory:3`, `roundHistory:task=ocr`,
    `roundHistory:teacher=ocr-early`, `roundHistory:stage=early`, and
    `roundHistory:round=2`
  - Supports chained and aggregate round-history queries such as
    `roundHistory:task=ocr:last`, `roundHistory:teacher=ocr-early:last`,
    `roundHistory:task=ocr:count`, and `roundHistory:task=ocr:meanLoss`
  - Supports compact scalar aliases and summary views such as
    `roundHistory:task=ocr:avgLoss`,
    `roundHistory:teacher=ocr-early:lastLoss`, and
    `roundHistory:task=ocr:summary`
  - Supports ranked queries such as
    `roundHistory:task=ocr:sort=-averageLoss:top=3` and
    `roundHistory:teacher=ocr-early:sort=round:top=5`
- Byte-latent checkpoint history inspection with
  `trainer_byte_latent_history_inspector.java`
  - Accepts either a checkpoint directory or direct `byte-latent-history.csv`
    path
  - Prefers `byte-latent-report.json` when present so one checkpoint entrypoint
    can query both summary and history artifacts
  - Loads byte-latent history through `ByteLatentHistoryReports`
  - Supports top-level sections such as `overview`, `summary`, `artifacts`,
    and `historyCount`
  - Supports nested summary queries such as `summary:metadata:globalStep`
  - Supports compact selectors such as `history:summary`,
    `history:lastLoss`, `history:lastEpoch`, `history:epoch=2`, and
    `history:sort=-trainLoss:top=3`
  - Supports `text`, pretty-printed `json`, and spreadsheet-ready `csv`
    output
  - Can infer export format from output filenames like `.json` and `.csv`
  - Supports optional export-to-file output paths as a fourth argument
- End-to-end byte-latent checkpoint generation with
  `trainer_byte_latent_demo.java`
  - Builds a tiny UTF-8 text dataset and byte-latent trainer config
  - Runs a deterministic 3-epoch byte-latent training session
  - Writes `byte-latent-summary.json`, `byte-latent-history.csv`,
    `byte-latent-report.json`, and `byte-latent-checkpoint.metadata`
  - Prints ready-to-run `trainer_byte_latent_history_inspector.java`
    commands against the generated checkpoint directory, including `status`,
    `health`, and `ci`
- End-to-end byte-latent checkpoint resume with
  `trainer_byte_latent_resume_demo.java`
  - Seeds a one-epoch byte-latent checkpoint directory
  - Resumes the same checkpoint with `resumeFromCheckpoint(true)` to a
    longer target epoch count
  - Prints resumed `historyRowCount`, `globalStep`, `resumeLoaded`, and
    follow-up inspector commands for `status`, `health`, `ci`, `overview`,
    `summary`, and `history:summary`
- One-command byte-latent smoke flow with
  `run_byte_latent_smoke.sh`
  - Runs `trainer_byte_latent_demo.java` into a checkpoint directory
  - Verifies `byte-latent-summary.json`, `byte-latent-history.csv`,
    `byte-latent-report.json`, and `byte-latent-checkpoint.metadata`
  - Runs the inspector against `status`, `health`, `ci`, and
    `history:sort=-trainLoss:top=3`
  - Prints the next-step resume command for `trainer_byte_latent_resume_demo.java`
  - Supports per-role overrides such as
    `GOLLEK_SD_NATIVE_MODEL_DIR_STUDENT` and
    `GOLLEK_SD_NATIVE_MODEL_DIR_OCR_EARLY`

## Prerequisites

From `gollek/` project root:

```bash
./gradlew publishJbangTrainerExamplesToMavenLocal
```

The task is Gradle-only and publishes the local snapshot runtime graph required
by the trainer JBang examples.

To validate the quality-profile CI gate evidence example end-to-end from
Gradle:

```bash
./gradlew smokeJbangTrainerRuntimeProfileBudgetGate
./gradlew verifyJbangTrainerRuntimeProfileBudgetGateOutput
./gradlew smokeJbangTrainerQualityProfileCiGateEvidence
./gradlew verifyJbangTrainerQualityProfileCiGateEvidenceOutput
./gradlew verifyJbangTrainerEvidenceIndex
./gradlew smokeJbangTrainerExamples
```

The runtime profile budget gate smoke task publishes the trainer JBang runtime
graph, runs `trainer_runtime_profile_budget_gate.java`, and verifies the
canonical runtime report plus JSON/Markdown/JUnit gate artifacts under
`build/jbang/trainer/runtime-profile-budget-gate`. The same example also writes
runtime input-profile gate artifacts under `runtime-input-profile-gate`, so a
dominant `train.next()` hotspot now produces concrete `DataLoader.prefetch(...)`
guidance in JSON/Markdown/JUnit evidence instead of only a generic runtime
warning. The verifier also writes `runtime-profile-budget-gate.summary.txt` for
compact CI uploads and quick human triage.

The quality-profile focused smoke task publishes the trainer JBang runtime graph, runs
`trainer_quality_profile_ci_gate_evidence.java`, requires the quality gate to
pass, and writes its output under
`build/jbang/trainer/quality-profile-ci-gate-evidence`. The verifier task then
checks the canonical reports, CI manifest, JSON verification report, Markdown
verification report, JUnit XML verification report, and compact
`quality-profile-ci-gate.summary.txt` CI summary.

The aggregate `verifyJbangTrainerEvidenceIndex` task checks both compact
summaries, verifies every referenced artifact still exists, and writes
`build/jbang/trainer/trainer-evidence-index.json` plus
`trainer-evidence-index.summary.txt` as a single release/CI handoff point. The
aggregate `smokeJbangTrainerExamples` task depends on this index, so the
verifier-backed trainer smoke lane now fails if either lane is missing evidence
or the combined index cannot be assembled. The Gradle lane uses an isolated
`build/jbang/.jbang` home and offline JBang resolution by default so dependency
refresh cannot silently hang CI.
