package tech.kayys.trellis2;

import tech.kayys.trellis2.pipelines.Trellis2ImageTo3DPipeline;
import tech.kayys.trellis2.pipelines.Trellis2ImageTo3DPipeline.*;
import tech.kayys.trellis2.models.FlowMatchingSampler.SamplerConfig;
import tech.kayys.trellis2.representations.Representations.Mesh;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * TRELLIS.2 Java — Main entry point
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE OVERVIEW
 * ════════════════════════════════════════════════════════════════════════════
 *
 * TRELLIS.2 is a 4B-parameter image-to-3D generation model by Microsoft.
 * This Java 25 port maps every Python module to an idiomatic Java equivalent:
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                        TRELLIS.2 Java Architecture                     │
 * │                                                                         │
 * │  Python module                   → Java 25 equivalent                  │
 * │  ─────────────────────────────────────────────────────────────────────  │
 * │  SparseTensor (feats, coords)     → SparseTensor record + FFM segments  │
 * │  nn.Module.forward()              → plain method, Arena param           │
 * │  torch.randn / torch.zeros        → Arena.allocate + fill               │
 * │  F.silu / F.sigmoid               → inline float math in Java           │
 * │  torch.utils.checkpoint           → (omitted — training only)           │
 * │  @dataclass / dict                → Java record                         │
 * │  str literal 'nearest'|'s2c'      → sealed interface hierarchy          │
 * │  safetensors.load_file()          → SafetensorsBridge (FFM mmap)        │
 * │  o_voxel C extension              → OVoxelBridge (FFM downcall)         │
 * │  nvdiffrast                       → NvdiffrastBridge (FFM downcall)     │
 * │  FlexGEMM (Triton kernel)         → FlexGemmBridge (FFM downcall)       │
 * │  pipeline.run(image)              → Trellis2ImageTo3DPipeline.run()     │
 * │                                                                         │
 * │  Key Java 25 features used:                                             │
 * │  ─────────────────────────────────────────────────────────────────────  │
 * │  • FFM API (JEP 454) — zero-copy native memory, downcall handles       │
 * │  • Sealed interfaces (JEP 409) — exhaustive pattern matching            │
 * │  • Records (JEP 395) — immutable data carriers                          │
 * │  • Pattern matching switch (JEP 441) — dispatch on sealed types        │
 * │  • Virtual threads (JEP 425) — per-stage isolation                     │
 * │  • StructuredTaskScope (JEP 453) — coordinated pipeline stages         │
 * │  • String templates / text blocks — logging + debug output              │
 * │                                                                         │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ════════════════════════════════════════════════════════════════════════════
 * PYTHON → JAVA MODULE MAPPING
 * ════════════════════════════════════════════════════════════════════════════
 *
 * trellis2/__init__.py
 *   → tech.kayys.trellis2 package
 *
 * trellis2/modules/sparse/__init__.py  (SparseTensor class)
 *   → modules/sparse/SparseTensor.java
 *   → modules/sparse/SparseConv3d.java
 *   → modules/sparse/SparseLinear.java
 *   → modules/sparse/SparseDownsample.java
 *   → modules/sparse/SparseUpsample.java
 *   → modules/sparse/SparseSpatial2Channel.java
 *   → modules/sparse/SparseChannel2Spatial.java
 *
 * trellis2/modules/sparse/attention/  (windowed + full sparse attention)
 *   → modules/sparse/attention/SparseWindowedAttention.java
 *   → modules/sparse/attention/SparseFullAttention.java
 *   → modules/sparse/attention/SparseRoPE.java
 *
 * trellis2/modules/sparse/transformer/
 *   → modules/sparse/transformer/SparseTransformerBlock.java
 *   → modules/sparse/transformer/SparseModulatedTransformerBlock.java
 *
 * trellis2/modules/attention/
 *   → modules/attention/MultiHeadAttention.java
 *   → modules/attention/RoPE.java
 *
 * trellis2/modules/transformer/
 *   → modules/transformer/TransformerBlock.java
 *   → modules/transformer/ModulatedTransformerBlock.java
 *
 * trellis2/modules/norm.py  (LayerNorm32)
 *   → modules/LayerNorm32.java
 *
 * trellis2/models/sc_vaes/sparse_unet_vae.py
 *   → models/scvaes/SparseResBlock3d.java         ✅ implemented
 *   → models/scvaes/SparseResBlockDownsample3d.java
 *   → models/scvaes/SparseResBlockUpsample3d.java
 *   → models/scvaes/SparseResBlockS2C3d.java
 *   → models/scvaes/SparseResBlockC2S3d.java
 *   → models/scvaes/SparseConvNeXtBlock3d.java
 *   → models/scvaes/SparseUnetVaeEncoder.java
 *   → models/scvaes/SparseUnetVaeDecoder.java
 *
 * trellis2/models/sc_vaes/fdg_vae.py
 *   → models/scvaes/FlexiDualGridVaeEncoder.java
 *   → models/scvaes/FlexiDualGridVaeDecoder.java
 *
 * trellis2/models/sparse_structure_vae.py
 *   → models/SparseStructureEncoder.java
 *   → models/SparseStructureDecoder.java
 *
 * trellis2/models/sparse_structure_flow.py
 *   → models/SparseStructureFlowModel.java
 *
 * trellis2/models/structured_latent_flow.py
 *   → models/SLatFlowModel.java
 *   → models/ElasticSLatFlowModel.java
 *
 * trellis2/models/sparse_elastic_mixin.py
 *   → models/SparseElasticMixin.java  (interface)
 *
 * trellis2/pipelines/__init__.py  (Trellis2ImageTo3DPipeline)
 *   → pipelines/Trellis2ImageTo3DPipeline.java    ✅ implemented
 *   → pipelines/Trellis2TexturingPipeline.java
 *
 * trellis2/renderers/  (EnvMap, render_video)
 *   → renderers/EnvMap.java
 *   → renderers/RenderUtils.java
 *   → renderers/NvdiffrastRenderer.java (FFM)
 *
 * o_voxel  (native C extension)
 *   → ffm/OVoxelBridge.java                       ✅ implemented
 *
 * FlexGEMM  (Triton/CUDA sparse conv)
 *   → ffm/FlexGemmBridge.java
 *
 * nvdiffrast, nvdiffrec_render  (rendering)
 *   → ffm/NvdiffrastBridge.java
 *   → ffm/NvdiffrecBridge.java
 *
 * ════════════════════════════════════════════════════════════════════════════
 * MEMORY MODEL
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Arena.ofShared()   — model weights (loaded once, freed on pipeline.close())
 *  Arena.ofConfined() — per-inference intermediates (freed after run())
 *  Arena.ofConfined() — per-step intermediates in sampler (freed each ODE step)
 *
 * ════════════════════════════════════════════════════════════════════════════
 */
public final class Trellis2Main {

    private static final System.Logger LOG =
            System.getLogger(Trellis2Main.class.getName());

    public static void main(String[] args) throws Exception {
        System.out.println("""
            ╔══════════════════════════════════════════════════════╗
            ║         TRELLIS.2 Java 25 — Image to 3D             ║
            ║  Port of microsoft/TRELLIS.2 using FFM API (JEP454) ║
            ╚══════════════════════════════════════════════════════╝
            """);

        String modelPath  = args.length > 0 ? args[0] : "microsoft/TRELLIS.2-4B";
        String imagePath  = args.length > 1 ? args[1] : "assets/example_image/T.png";
        String outputPath = args.length > 2 ? args[2] : "output.glb";
        String resolution = args.length > 3 ? args[3] : "512";

        // ── Load pipeline (model weights via SafeTensors FFM bridge) ──────────
        LOG.log(System.Logger.Level.INFO, "Loading pipeline from: {0}", modelPath);
        try (var pipeline = Trellis2ImageTo3DPipeline.fromPretrained(modelPath)) {

            // ── Load and preprocess image ─────────────────────────────────────
            byte[] imageBytes = Files.readAllBytes(Path.of(imagePath));
            byte[] processed  = pipeline.preprocessImage(imageBytes);
            LOG.log(System.Logger.Level.INFO, "Image preprocessed: {0}", imagePath);

            // ── Configure run params ──────────────────────────────────────────
            RunParams params = RunParams.builder()
                    .seed(42L)
                    .resolution(resolution)
                    .ssParams(new SamplerConfig(12, 7.5f, 0.7f, 5.0f))
                    .shapeSlatParams(new SamplerConfig(12, 7.5f, 0.5f, 3.0f))
                    .texSlatParams(new SamplerConfig(12, 1.0f, 0.0f, 3.0f))
                    .returnLatent(false)
                    .build();

            // ── Run pipeline ──────────────────────────────────────────────────
            LOG.log(System.Logger.Level.INFO, "Running TRELLIS.2 pipeline (resolution={0})...", resolution);
            long t0 = System.currentTimeMillis();

            RunResult result = pipeline.run(processed, params);

            long elapsed = System.currentTimeMillis() - t0;
            LOG.log(System.Logger.Level.INFO,
                    "Pipeline completed in {0}ms", elapsed);

            // ── Export to GLB ─────────────────────────────────────────────────
            List<Mesh> meshes = result.meshes();
            if (!meshes.isEmpty()) {
                Mesh mesh = meshes.getFirst();
                mesh.simplify(16_777_216); // nvdiffrast face limit
                mesh.exportGlb(Path.of(outputPath), 100_000, 2048);
                LOG.log(System.Logger.Level.INFO,
                        "Exported GLB: {0} (vertices={1}, faces={2})",
                        outputPath, mesh.vertexCount(), mesh.faceCount());
            }

            System.out.println("""

                ✓ Done!
                  Input:   %s
                  Output:  %s
                  Time:    %dms
                  Meshes:  %d
                """.formatted(imagePath, outputPath, elapsed, meshes.size()));
        }
    }
}
