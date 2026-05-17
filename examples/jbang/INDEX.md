# Gollek JBang Examples

This folder is the runnable JBang surface for Gollek demos and integration
playbooks. It tracks the Gradle-first module vision (`ml/*`, `trainer/*`,
`integration/*`) while keeping older SDK-named scripts for compatibility.

## Quick Start

1. Install JBang:
   - `brew install jbang/tap/jbang`
2. Prepare local artifacts from `gollek/` project root:
   - `./run-install-local-macos.sh`
3. Run examples from `gollek/examples/jbang`:
   - `jbang trainer/trainer_runtime_bootstrap.java`
   - `jbang sdk/gollek-quickstart.java`
   - `jbang sdk/gollek-sdk-core-example.java`
   - `jbang sdk/gollek-sdk-train-example.java 2`
   - `jbang sdk/gollek-sdk-vision-example.java`
   - `jbang sdk/gollek-sdk-export-example.java`
   - `jbang sdk/gollek-sdk-augment-example.java`

## Recommended Paths

### Trainer Runtime (Canonical)

- `trainer/trainer_runtime_bootstrap.java`
  - Uses `gollek-trainer` + `gollek-trainer-api`
  - Prints runtime mode (`legacy-bridge` vs `canonical-fallback`)
  - Shows listener lifecycle and training summary
- `trainer/trainer_diffusion_opd_ddim.java`
  - Uses `gollek-ml-api` + `gollek-ml-diffusion-opd` + `gollek-diffusion`
  - Demonstrates Java-first DiffusionOPD wiring with DDIM-style scheduler flow
  - Shows task prompts, teacher adapters, and `Gollek.DL.diffusionOpdTrainer()`

### SDK Compatibility Scripts (Runnable Today)

- `sdk/gollek-quickstart.java`
- `sdk/gollek-sdk-core-example.java`
- `sdk/gollek-sdk-train-example.java`
- `sdk/gollek-sdk-vision-example.java`
- `sdk/gollek-sdk-export-example.java`
- `sdk/gollek-sdk-augment-example.java`

These are compatibility entrypoints with legacy file names, but they now run
against the current local artifact set.

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
| `integration/` | 3rd-party integrations | DL4J, Smile, Tribuo, OpenNLP |
| `common/` | Baseline utilities | Legacy starter scripts |

## Compatibility Notes

- Scripts named `gollek-sdk-*` are compatibility examples, not canonical naming.
- Some older examples depend on `${user.home}/.gollek/jbang/libs/*` local jars.
- Prefer scripts that use `//REPOS local,mavencentral` plus
  targeted `publishToMavenLocal` tasks.
