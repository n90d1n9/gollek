# Gollek JBang Examples

This folder is the runnable JBang surface for Gollek demos and integration
playbooks. It tracks the Gradle-first module vision (`ml/*`, `trainer/*`,
`integration/*`) while keeping older SDK-named scripts for compatibility.

## Quick Start

1. Install JBang:
   - `brew install jbang/tap/jbang`
2. Prepare local artifacts from `gollek/` project root:
   - `./run-install-local-macos.sh`
   - `./gradlew publishJbangTrainerExamplesToMavenLocal`
   - `./gradlew smokeJbangTrainerRuntimeProfileBudgetGate`
   - `./gradlew verifyJbangTrainerRuntimeProfileBudgetGateOutput`
   - `./gradlew smokeJbangTrainerQualityProfileCiGateEvidence`
   - `./gradlew verifyJbangTrainerQualityProfileCiGateEvidenceOutput`
   - `./gradlew smokeJbangTrainerExamples`
   - `./gradlew smokeJbangAgentBridgeExamples -Pgollek.jbang.agentBridge.offline=false`
   - The Gradle smoke lane uses `build/jbang/.jbang` plus offline JBang
     resolution after the local artifacts are published. Add
     `-Pgollek.jbang.trainer.offline=false` only to refresh remote JBang
     dependencies intentionally.
3. Run examples from `gollek/examples/jbang`:
   - `jbang trainer/trainer_runtime_bootstrap.java`
   - `jbang trainer/trainer_runtime_profile_budget_gate.java`
   - `jbang trainer/trainer_quality_profile_ci_gate_evidence.java`
   - `jbang sdk/gollek-quickstart.java`
   - `jbang sdk/gollek-sdk-core-example.java`
   - `jbang sdk/gollek-sdk-train-example.java 2`
   - `jbang sdk/gollek-sdk-vision-example.java`
   - `jbang sdk/gollek-sdk-export-example.java`
   - `jbang sdk/gollek-sdk-augment-example.java`
   - `jbang run --no-integrations integration/wayang_gollek_serving_bridge.java --mock`

## Recommended Paths

### Trainer Runtime (Canonical)

- `trainer/trainer_runtime_bootstrap.java`
  - Uses `gollek-trainer` + `gollek-trainer-api`
  - Prints runtime mode (`legacy-bridge` vs `canonical-fallback`)
  - Shows listener lifecycle and training summary
- `trainer/trainer_runtime_profile_budget_gate.java`
  - Uses `gollek-ml-api`
  - Builds a tiny canonical runtime-profile report, evaluates strict trainer
    runtime budgets, writes JSON/Markdown/JUnit evidence, and verifies the
    persisted artifact bundle
  - Supports `--out`, named policies, threshold overrides, and `--fail-on-gate`
- `trainer/trainer_quality_profile_ci_gate_evidence.java`
  - Uses `gollek-ml-api` to train tiny baseline/candidate reports
  - Runs a quality-profile CI gate, writes the manifest, writes
    JSON/Markdown/JUnit verification evidence, and verifies the persisted
    evidence bundle
  - Supports `--out`, `--device`, `--profile`, epoch controls, and
    `--fail-on-gate`
- `trainer/trainer_diffusion_opd_ddim.java`
  - Uses `gollek-ml-api` + `gollek-ml-diffusion-opd` + `gollek-diffusion`
  - Demonstrates Java-first DiffusionOPD wiring with DDIM-style scheduler flow
  - Shows task prompts, teacher adapters, and `Gollek.DL.diffusionOpdTrainer()`
- `trainer/trainer_diffusion_opd_stable_diffusion_metal_bridge.java`
  - Uses `gollek-ml-diffusion-opd` + `gollek-backend-metal` +
    `gollek-runner-stable-diffusion`
  - Demonstrates the Metal-aware stable-diffusion bridge round-trip path
  - Shows accelerator metadata and scheduler alignment for OPD integration
- `trainer/trainer_diffusion_opd_stable_diffusion_native_bridge.java`
  - Uses `gollek-ml-api` + `gollek-ml-diffusion-opd` +
    `gollek-runner-stable-diffusion`
  - Demonstrates end-to-end OPD wiring with stable-diffusion-native UNet and
    executable PNDM scheduler bridge surfaces together
  - Shows task routing, adaptive stage weighting, trainer artifact output, and
    real CLIP prompt conditioning when `text_encoder/` and `tokenizer/` are
    present
  - Supports task-specific CLIP fixture overrides for `ocr` and `aesthetics`
- `trainer/trainer_diffusion_opd_report_inspector.java`
  - Uses `gollek-ml-diffusion-opd`
  - Loads `diffusion-opd-report.json` and prints either the full report or a
    selected section such as `overview`, `conditioning`, `teachers`, or
    `roundHistory`, in `text`, pretty-printed `json`, `table`, or `csv`
  - Supports grouped rollups like `taskSummaries` and `teacherSummaries`
  - Supports ranked grouped rollups like `taskSummaries:sort=-meanLoss:top=5`
  - Supports grouped pair rollups like
    `taskTeacherSummaries:sort=-meanLoss:top=10`
  - Supports export column projection like `value,count,meanLoss`
  - Supports optional export-to-file output paths for `text`, `json`, `table`,
    and `csv`; `_` remains available as a no-projection placeholder when
    needed, but most `json`/`csv` cases can now use filename inference
  - Can infer export format from output filenames like `.json`, `.csv`,
    `.table`, `.md`, `.txt`, or `.log`
  - Supports directory-oriented bundle export with `bundle=standard` and
    `bundle=rollups`
  - Supports custom bundle definitions like
    `bundle=custom:overview,taskSummaries,taskTeacherSummaries`
  - Custom bundles support inline aliases like
    `bundle=custom:overview@run-overview,taskTeacherSummaries@teacher-pairs`
  - Custom bundles support per-entry columns like
    `bundle=custom:taskSummaries@tasks#compact,taskTeacherSummaries@pairs#compare`
  - Custom bundles support per-entry formats like
    `bundle=custom:overview@run-overview!json,taskTeacherSummaries@teacher-pairs!csv`
  - Bundle directories include a `manifest.json` index with timestamps and
    per-file section metadata for downstream tooling
  - Bundle directories can be inspected directly with `manifest`,
    `bundleSummary`, `bundleHealth`, `files`, `filesSummary:by=section`,
    `filesSummary:by=format:sort=-count`,
    `filesSummary:by=section:sort=-count:top=3`,
    `files:format=csv`, `files:section=taskTeacherSummaries`, or
    `file:<name>` and `loadfile:...` views, with
    `sort=name|-name|section|count`, `top=<n>`, `pick=last`, or `index=<n>`
    selectors for multi-match cases
  - `bundleSummary` supports `focus=sections|formats|all` and `top=<n>` for
    compact rollup views, plus `dominant=true` for headline-only output
  - `bundleSummary` also reports missing bundle files referenced by the
    manifest
  - `bundleHealth` gives a compact pass/fail integrity view and groups
    missing files by section and format, with normalized `status`,
    `healthScore`, `alertLevel`, stable `issueCodes`, compact `healthBadge`
    with UI-ready `variant`, `label`, `token`, `tooltip`, `checkStatus`,
    `primaryCheckCode`, and `primaryCheckSeverity`,
    and human-facing `summaryMessage` / `recommendedAction`, plus structured named checks,
    normalized `checkSummary` including pass/failure rates, local status/alert fields, local guidance strings, local issue codes, local primary issue code, a nested badge with its own primary issue code, primary check name/message/severity, summary/action guidance, token/tooltip/check-status support, and direct action hints, `allPassed`, fast critical/warning/info failure flags, failing severity count breakdowns, primary failing check identity/severity/message, compact failing names/codes, compact failing-check summaries, and aggregate
    check counters
  - `bundleHealth` supports `focus=files|sections|formats|all` and `top=<n>`
    for compact diagnosis slices
  - `table`/`csv` bundles use section-aware default projections like
    `compact` for grouped summaries and `compare` for pair rollups
  - Supports alias columns like `loss`, `latestRound`, `latestLoss`,
    `top1Round`, `top1Loss`, `first1Round`, and `first1Loss`
  - Supports projection presets like `minimal`, `compact`, `latest`,
    `leaderboard`, `details`, and `compare`
  - `details` and `compare` can flatten both top-loss and first-round
    representative rows into export-friendly columns
  - Presets adapt to row shape, using `pair` for pair rollups and `value` for
    grouped single-dimension rollups
  - Supports compact presets like `taskSummary=ocr` and
    `teacherSummary=ocr-early`
  - Supports pair presets like `taskTeacherSummary=ocr,ocr-early`
    and `taskStageSummary=ocr,early`
  - Supports focused selectors like `roundHistory:last` and
    `roundHistory:task=ocr`
  - Supports compact queries like `roundHistory:task=ocr:last` and
    `roundHistory:task=ocr:meanLoss`
  - Supports scalar aliases and summary rollups like
    `roundHistory:teacher=ocr-early:lastLoss` and
    `roundHistory:task=ocr:summary`
  - Supports ranked queries like
    `roundHistory:task=ocr:sort=-averageLoss:top=3`
- `trainer/trainer_byte_latent_history_inspector.java`
  - Uses `gollek-ml-byte-latent`
  - Accepts either a byte-latent checkpoint directory or direct
    `byte-latent-history.csv` path
  - Supports `status` as the shortest checkpoint-side status selector
  - Supports `ci` as the shortest compact CI-style checkpoint status selector
  - Supports `status:ci` as the explicit CI-style checkpoint status selector
  - Supports compact `history:` selectors like `history:summary`,
    `history:lastLoss`, `history:epoch=2`, and
    `history:sort=-trainLoss:top=3`
  - Supports `text`, pretty-printed `json`, and spreadsheet-ready `csv`
    output, with optional export-to-file path as a fourth argument
- `trainer/trainer_byte_latent_demo.java`
  - Uses `gollek-ml-byte-latent`
  - Trains a small byte-latent demo run from UTF-8 text lines
  - Writes `byte-latent-summary.json`, `byte-latent-history.csv`,
    `byte-latent-report.json`, and the checkpoint manifest
  - Prints ready-to-run follow-up inspector commands for the generated
    checkpoint directory
- `trainer/trainer_byte_latent_resume_demo.java`
  - Uses `gollek-ml-byte-latent`
  - Seeds a one-epoch byte-latent checkpoint and then resumes it to a longer
    target epoch count
  - Shows resumed history growth, loaded resume metadata, and continued
    report/checkpoint inspection commands
- `trainer/trainer_byte_latent_infer_demo.java`
  - Uses `gollek-ml-byte-latent`
  - Runs a tiny reference byte-latent prompt continuation flow
  - Prints prompt token ids, `predictNextToken(...)`, generated token ids, and
    combined byte/text output for a short deterministic continuation
- `trainer/trainer_byte_latent_train_infer_demo.java`
  - Uses `gollek-ml-byte-latent`
  - Runs a tiny trainer session and then immediately performs prompt continuation
  - Prints training summary, checkpoint paths, next-token prediction, and short
    deterministic generation from the same model family/spec
  - Writes `train-infer-report.json` with both training and inference sections
- `trainer/trainer_byte_latent_train_infer_inspector.java`
  - Uses Jackson only and reads `train-infer-report.json`
  - Supports `overview`, `training`, `inference`, `nextToken`,
    `generatedText`, `combinedText`, `status`, and `ci`
- `trainer/trainer_byte_latent_workflow_inspector.java`
  - Uses Jackson only and reads `byte_latent_workflow_manifest.json`
  - Supports `overview`, `runs`, `workflowBundleDir`,
    `workflowBundleStatus`, `run:fresh`, `historyRows`,
    `delta:historyRows`, and `delta:latestTrainLoss`
  - Supports `bundleOverview` and
    `bundleOverview:focus=health|files|all` plus `summary=short` for compact
    bundle summaries
  - Supports `bundleOverview:ci` as the shorthand alias for the recommended
    CI-style overview check
  - Supports `status` as the shortest neutral bundle check selector, with
    `bundleStatus` kept as the explicit form
  - Supports `bundleHealth:ci` and `bundleHealth:summary=short` for single-line
    direct health checks
  - Can jump directly from a workflow manifest into a materialized bundle with
    `bundleManifest`, `bundleFiles`, `bundleSummary`, `bundleHealth`, and
    `bundleLoadfile:section=<section>`
  - Supports `bundle=standard` to export overview, runs, and delta views into
    one output directory with a small bundle manifest
  - Can inspect exported bundle directories with `bundleSummary`,
    `bundleHealth`, `files`, `files:section=<section>`,
    `files:name=<name>`, `file:<name>`, and `loadfile:section=<section>`
  - Supports `files:...:sort=name|section|format` for compact ordered listings
  - Supports `loadfile:section=<section>:pick=first|last` and `index=<n>`
  - Supports `text`, pretty-printed `json`, and spreadsheet-ready `csv`
- `trainer/run_byte_latent_smoke.sh`
  - Runs the byte-latent demo plus key inspector checks in one command
  - Verifies the expected checkpoint artifacts exist
  - Prints follow-up commands for resumed-run inspection
- `trainer/run_byte_latent_workflow.sh`
  - Unifies fresh-run, resumed-run, and full smoke flows behind one entrypoint
  - Supports `fresh`, `resume`, and `full` modes plus optional checkpoint-root
  - Prints a compact final workflow summary with checkpoint paths, history row
    counts, and inspector-ready commands
  - Writes a machine-readable workflow manifest JSON for CI or follow-up tools
  - Records `workflowBundleDir` and `workflowBundleStatus` in that manifest
  - Materializes a workflow bundle automatically in `full` mode so the printed
    `loadfile:` examples point at a ready directory
  - Prints ready-to-run workflow-manifest inspector commands at the end
  - Delegates to the existing smoke helpers so older commands keep working
- `trainer/run_byte_latent_resume_smoke.sh`
  - Runs the byte-latent resume demo plus key inspector checks in one command
  - Verifies the resumed checkpoint artifacts and persisted history growth
  - Prints follow-up commands for continued checkpoint inspection

### SDK Compatibility Scripts (Runnable Today)

- `sdk/gollek-quickstart.java`
- `sdk/gollek-sdk-core-example.java`
- `sdk/gollek-sdk-train-example.java`
- `sdk/gollek-sdk-vision-example.java`
- `sdk/gollek-sdk-export-example.java`
- `sdk/gollek-sdk-augment-example.java`

These are compatibility entrypoints with legacy file names, but they now run
against the current local artifact set.

### Agent Serving Bridge

- `integration/wayang_gollek_serving_bridge.java`
  - Uses the lightweight `gollek-sdk-agent` module directly
  - The full `gollek-sdk-remote` client still exposes the same agent client
    through `client.agent()`
  - Demonstrates Gollek as the serving/inference engine for Wayang-Gollek
  - Shows capability and contract discovery, MCP tool schema discovery,
    request validation, embeddings, caller-owned RAG retrieval, streamed chat
    deltas, streamed tool-call previews, tool-call handoff, and Wayang-owned
    memory
  - Runs with `--mock` by default so the flow can be inspected without a live
    server; use `jbang run --no-integrations ...` because this plain Java
    bridge does not need JBang's Quarkus post-build integration
  - CI/local smoke: `./gradlew smokeJbangAgentBridgeExamples
    -Pgollek.jbang.agentBridge.offline=false`
  - Smoke output is written to
    `build/jbang/integration/wayang-gollek-serving-bridge.out`
  - Use `--live --base-url http://localhost:8080 --api-key community` against
    a running Gollek endpoint after publishing
    `:sdk:gollek-sdk-agent:publishToMavenLocal`

### Experimental v0.2/v0.3 Scripts

- `sdk/tensor_operations_v02.java`
- `sdk/vision_transforms_v02.java`
- `sdk/tokenization_v02.java`
- `sdk/mnist_training_v02.java`
- `sdk/pytorch_comparison_v02.java`
- `sdk/unified_framework_demo.java`
- `sdk/graph_fusion_example.java`

These are useful references, but dependency coordinates vary and may require
extra local publishing beyond the quickstart path.

### Inference and Modality Demos

- `nlp/*`
- `multimodal/*`
- `edge/*`
- `quantizer/*`
- `integration/*`

## Directory Map

| Folder | Focus | Notes |
|---|---|---|
| `trainer/` | Canonical trainer runtime | Aligned with `:trainer:*` modules |
| `sdk/` | ML framework and compatibility examples | Mixed canonical + legacy naming |
| `nlp/` | NLP pipelines and chat demos | Includes SIMD and GGUF flows |
| `multimodal/` | Vision/audio/text demos | Mixed quality, evolving APIs |
| `edge/` | LiteRT and edge paths | Device-oriented scenarios |
| `quantizer/` | Quantization experiments | AWQ/GPTQ/TurboQuant samples |
| `integration/` | 3rd-party integrations and agent bridges | Wayang-Gollek, DL4J, Smile, Tribuo, OpenNLP |
| `common/` | Baseline utilities | Legacy starter scripts |

## Compatibility Notes

- Scripts named `gollek-sdk-*` are compatibility examples, not canonical naming.
- Some older examples depend on `${user.home}/.gollek/jbang/libs/*` local jars.
- Prefer scripts that use `//REPOS local,mavencentral` plus
  targeted `publishToMavenLocal` tasks.
