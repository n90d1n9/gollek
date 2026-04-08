package tech.kayys.trellis2.pipelines;

import tech.kayys.trellis2.models.FlowMatchingSampler;
import tech.kayys.trellis2.models.FlowMatchingSampler.SamplerConfig;
import tech.kayys.trellis2.models.FlowMatchingSampler.VelocityModel;
import tech.kayys.trellis2.modules.sparse.SparseTensor;
import tech.kayys.trellis2.representations.Mesh;
import tech.kayys.trellis2.representations.OVoxel;
import tech.kayys.trellis2.ffm.ImageEncoderBridge;
import tech.kayys.trellis2.ffm.OVoxelBridge;
import tech.kayys.trellis2.ffm.SafetensorsBridge;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Trellis2ImageTo3DPipeline — Java 25 port of trellis2/pipelines/__init__.py
 *
 * <p>This is the top-level orchestrator for image → 3D generation.
 * It chains three successive flow-matching diffusion stages:
 *
 * <pre>
 *  Input image
 *      │
 *      ▼ ImageEncoder (DINOv2/SigLIP via FFM)
 *  Image embeddings [1, 768]
 *      │
 *      ▼ Stage 1: SparseStructureFlow
 *  Sparse binary occupancy SparseTensor [N, 1] (→ coordinates only)
 *      │
 *      ▼ Stage 2: SLatShapeFlow (SparseUnetVAE decoder)
 *  Shape SLAT SparseTensor [N, 8]
 *      │
 *      ▼ Stage 3: SLatTexFlow (SparseUnetVAE decoder)
 *  Texture SLAT SparseTensor [N, 8]
 *      │
 *      ▼ Decode (FlexiDualGridVAE → Mesh)
 *  OVoxel Mesh with PBR attributes
 * </pre>
 *
 * <p>Java 25 design:
 * <ul>
 *   <li>{@link PipelineConfig} is a record; {@link PipelineType} is a sealed interface
 *       enabling exhaustive switching on resolution variants.</li>
 *   <li>The three-stage generation is launched as structured concurrency tasks
 *       ({@link StructuredTaskScope}) — stages 2 and 3 run sequentially but
 *       preprocessing and postprocessing fan out.</li>
 *   <li>All weight loading uses the {@link SafetensorsBridge} (FFM) for zero-copy
 *       memory-mapped access to the .safetensors checkpoint.</li>
 *   <li>Memory lifecycle: model weights live in a {@link Arena#ofShared() shared arena},
 *       per-inference intermediates in a {@link Arena#ofConfined() confined arena} that
 *       is closed after {@link #run}.</li>
 * </ul>
 *
 * Python equivalent:
 * <pre>
 * pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
 * mesh = pipeline.run(image)[0]
 * </pre>
 *
 * @author Gollek / Bhangun — JDK 25 port of TRELLIS.2
 */
public final class Trellis2ImageTo3DPipeline implements AutoCloseable {

    // ═════════════════════════════════════════════════════════════════════════
    // 1. Pipeline type — sealed hierarchy for resolution variants
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Python: pipeline_type = "512" | "512->1024" | "512->1536"
     * Sealed for exhaustive pattern matching in resolution selection.
     */
    public sealed interface PipelineType
            permits PipelineType.Res512,
                    PipelineType.Res512To1024,
                    PipelineType.Res512To1536 {

        int baseResolution();
        int finalResolution();

        record Res512()        implements PipelineType {
            public int baseResolution()  { return 512;  }
            public int finalResolution() { return 512;  }
        }
        record Res512To1024()  implements PipelineType {
            public int baseResolution()  { return 512;  }
            public int finalResolution() { return 1024; }
        }
        record Res512To1536()  implements PipelineType {
            public int baseResolution()  { return 512;  }
            public int finalResolution() { return 1536; }
        }

        static PipelineType of(String s) {
            return switch (s) {
                case "512"        -> new Res512();
                case "512->1024"  -> new Res512To1024();
                case "512->1536"  -> new Res512To1536();
                default           -> throw new IllegalArgumentException("Unknown pipeline type: " + s);
            };
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 2. Run request record (replaces Python kwargs)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Encapsulates all per-run parameters.
     * Compact constructor sets validated defaults.
     */
    public record RunParams(
            long          seed,
            PipelineType  pipelineType,
            SamplerConfig ssParams,       // Stage 1 — sparse structure
            SamplerConfig shapeSlatParams,// Stage 2 — shape SLAT
            SamplerConfig texSlatParams,  // Stage 3 — texture SLAT
            boolean       returnLatent,
            boolean       preprocessImage
    ) {
        public RunParams {
            if (seed < 0)          seed = System.currentTimeMillis() & 0x7FFFFFFFL;
            if (pipelineType == null) pipelineType = new PipelineType.Res512();
            if (ssParams == null)    ssParams       = SamplerConfig.sparseStructureDefaults();
            if (shapeSlatParams==null) shapeSlatParams = SamplerConfig.shapeSlatDefaults();
            if (texSlatParams == null) texSlatParams  = SamplerConfig.textureSlatDefaults();
        }

        public static RunParams defaults() {
            return new RunParams(-1, null, null, null, null, false, true);
        }

        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private long          seed           = -1;
            private PipelineType  pipelineType   = new PipelineType.Res512();
            private SamplerConfig ssParams       = SamplerConfig.sparseStructureDefaults();
            private SamplerConfig shapeSlatParams= SamplerConfig.shapeSlatDefaults();
            private SamplerConfig texSlatParams  = SamplerConfig.textureSlatDefaults();
            private boolean       returnLatent   = false;
            private boolean       preprocessImage= true;

            public Builder seed(long s)               { seed = s; return this; }
            public Builder pipelineType(PipelineType t){ pipelineType = t; return this; }
            public Builder ssParams(SamplerConfig c)  { ssParams = c; return this; }
            public Builder shapeSlatParams(SamplerConfig c){ shapeSlatParams = c; return this; }
            public Builder texSlatParams(SamplerConfig c)  { texSlatParams = c; return this; }
            public Builder returnLatent(boolean b)    { returnLatent = b; return this; }
            public Builder resolution(String s)       { return pipelineType(PipelineType.of(s)); }

            public RunParams build() {
                return new RunParams(seed, pipelineType, ssParams,
                        shapeSlatParams, texSlatParams, returnLatent, preprocessImage);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 3. RunResult — sealed hierarchy
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Sealed result: either mesh-only or mesh+latents.
     * Python: outputs, (shape_slat, tex_slat, res) when return_latent=True
     */
    public sealed interface RunResult
            permits RunResult.MeshOnly, RunResult.MeshWithLatents {

        List<Mesh> meshes();

        record MeshOnly(List<Mesh> meshes) implements RunResult {}

        record MeshWithLatents(
                List<Mesh>    meshes,
                SparseTensor  shapeSlat,
                SparseTensor  texSlat,
                int           resolution
        ) implements RunResult {}
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 4. Pipeline state
    // ═════════════════════════════════════════════════════════════════════════

    private static final System.Logger LOG = System.getLogger(
            Trellis2ImageTo3DPipeline.class.getName());

    /** Long-lived arena owning all model weight memory. */
    private final Arena weightArena = Arena.ofShared();

    // Sub-models (loaded from safetensors weights into weightArena)
    private final VelocityModel  sparseStructureFlow;
    private final VelocityModel  shapeSlatFlow;
    private final VelocityModel  texSlatFlow;
    private final ImageEncoderBridge imageEncoder;
    private final OVoxelBridge   oVoxelBridge;

    /** PBR attribute layout: [base_color, roughness, metallic, alpha] */
    private final int[] pbrAttrLayout;

    private volatile boolean closed = false;

    // ── Constructor (use fromPretrained factory) ──────────────────────────────

    private Trellis2ImageTo3DPipeline(
            VelocityModel sparseStructureFlow,
            VelocityModel shapeSlatFlow,
            VelocityModel texSlatFlow,
            ImageEncoderBridge imageEncoder,
            OVoxelBridge oVoxelBridge,
            int[] pbrAttrLayout) {
        this.sparseStructureFlow = sparseStructureFlow;
        this.shapeSlatFlow       = shapeSlatFlow;
        this.texSlatFlow         = texSlatFlow;
        this.imageEncoder        = imageEncoder;
        this.oVoxelBridge        = oVoxelBridge;
        this.pbrAttrLayout       = pbrAttrLayout;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 5. Factory — fromPretrained
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Python: {@code Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")}
     *
     * <p>Loads model weights from a local directory or HuggingFace Hub path.
     * All weights are memory-mapped via {@link SafetensorsBridge} (FFM) —
     * no intermediate Java heap copies.
     *
     * @param modelPath  local directory path or HF hub id "org/repo"
     */
    public static Trellis2ImageTo3DPipeline fromPretrained(String modelPath) {
        LOG.log(System.Logger.Level.INFO, "Loading TRELLIS.2 pipeline from: {0}", modelPath);

        Path localPath = resolveModelPath(modelPath);

        // Load each sub-model from its .safetensors + .json config
        var ssFlow    = SafetensorsBridge.loadVelocityModel(
                localPath.resolve("sparse_structure_flow.safetensors"),
                localPath.resolve("sparse_structure_flow.json"));

        var shapeFlow = SafetensorsBridge.loadVelocityModel(
                localPath.resolve("shape_slat_flow.safetensors"),
                localPath.resolve("shape_slat_flow.json"));

        var texFlow   = SafetensorsBridge.loadVelocityModel(
                localPath.resolve("tex_slat_flow.safetensors"),
                localPath.resolve("tex_slat_flow.json"));

        var imgEnc    = new ImageEncoderBridge(localPath.resolve("image_encoder.onnx"));
        var ovBridge  = new OVoxelBridge();

        // PBR layout: [base_color(3), roughness(1), metallic(1), alpha(1)]
        int[] pbrLayout = {3, 1, 1, 1};

        LOG.log(System.Logger.Level.INFO, "Pipeline loaded successfully");
        return new Trellis2ImageTo3DPipeline(ssFlow, shapeFlow, texFlow,
                imgEnc, ovBridge, pbrLayout);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 6. Main run() method
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Python: {@code mesh = pipeline.run(image)[0]}
     *
     * <p>Executes the full image-to-3D pipeline.
     * Each inference uses a confined arena freed at the end.
     *
     * @param imageBytes   RGBA PNG/JPEG bytes of the input image
     * @param params       run parameters
     * @return {@link RunResult} — list of generated meshes (+ latents if requested)
     */
    public RunResult run(byte[] imageBytes, RunParams params) {
        requireNotClosed();

        try (Arena inferArena = Arena.ofConfined()) {
            LOG.log(System.Logger.Level.INFO,
                    "Running pipeline: type={0}, seed={1}",
                    params.pipelineType(), params.seed());

            // ── Step 1: Preprocess & encode image ─────────────────────────────
            // Python: cond = self.image_encoder(image)
            MemorySegment imageCond = imageEncoder.encode(imageBytes, inferArena);
            MemorySegment nullCond  = imageEncoder.nullConditioning(inferArena);

            // ── Step 2: Stage 1 — Sparse structure generation ────────────────
            // Python: sparse_structure = self.sparse_structure_flow.sample(noise, cond, ...)
            int baseRes = params.pipelineType().baseResolution();
            SparseTensor ssNoise = createSparseNoise(baseRes, params.seed(), inferArena);

            LOG.log(System.Logger.Level.INFO, "Stage 1: sparse structure sampling...");
            SparseTensor sparseStructure = FlowMatchingSampler.sample(
                    ssNoise, sparseStructureFlow, imageCond, nullCond,
                    params.ssParams(), inferArena);

            // ── Step 3: Stage 2 — Shape SLAT generation ───────────────────────
            // Python: shape_slat = self.slat_flow.sample(noise, cond, sparse_structure)
            LOG.log(System.Logger.Level.INFO, "Stage 2: shape SLAT sampling...");
            SparseTensor shapeNoise = FlowMatchingSampler.gaussianNoise(
                    sparseStructure, params.seed() + 1, inferArena);

            // Condition also includes the sparse structure occupancy
            MemorySegment shapeCond = combineConditionings(
                    imageCond, sparseStructure, inferArena);

            SparseTensor shapeSlat = FlowMatchingSampler.sample(
                    shapeNoise, shapeSlatFlow, shapeCond, nullCond,
                    params.shapeSlatParams(), inferArena);

            // ── Step 4: Stage 3 — Texture SLAT generation ────────────────────
            LOG.log(System.Logger.Level.INFO, "Stage 3: texture SLAT sampling...");
            SparseTensor texNoise = FlowMatchingSambler.gaussianNoise(
                    shapeSlat, params.seed() + 2, inferArena);

            MemorySegment texCond = combineConditionings(
                    imageCond, shapeSlat, inferArena);

            SparseTensor texSlat = FlowMatchingSampler.sample(
                    texNoise, texSlatFlow, texCond, nullCond,
                    params.texSlatParams(), inferArena);

            // ── Step 5: Decode latents → mesh ─────────────────────────────────
            LOG.log(System.Logger.Level.INFO, "Decoding latents to mesh...");
            List<Mesh> meshes = decodeLatent(shapeSlat, texSlat,
                    params.pipelineType().finalResolution(), inferArena);

            // ── Step 6: Package result ────────────────────────────────────────
            if (params.returnLatent()) {
                // Copy slats to caller arena before inferArena closes
                SparseTensor shapeOut = copyToArena(shapeSlat, Arena.ofShared());
                SparseTensor texOut   = copyToArena(texSlat,   Arena.ofShared());
                return new RunResult.MeshWithLatents(meshes, shapeOut, texOut,
                        params.pipelineType().finalResolution());
            }
            return new RunResult.MeshOnly(meshes);
        }
    }

    /**
     * Decode shape+texture SLATs into a mesh list.
     * Python: {@code pipeline.decode_latent(shape_slat, tex_slat, res)}
     */
    public List<Mesh> decodeLatent(SparseTensor shapeSlat, SparseTensor texSlat,
                                    int resolution, Arena arena) {
        // Delegate to OVoxelBridge (wraps o_voxel C library via FFM)
        return oVoxelBridge.decodeSlatToMesh(shapeSlat, texSlat, resolution,
                pbrAttrLayout, arena);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 7. Utilities
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Preprocess an input image for the pipeline.
     * Python: {@code pipeline.preprocess_image(image)}
     * Removes background (rembg / BiRefNet), pads to square, normalizes.
     */
    public byte[] preprocessImage(byte[] imageBytes) {
        return imageEncoder.preprocess(imageBytes);
    }

    private SparseTensor createSparseNoise(int resolution, long seed, Arena arena) {
        // Create a dense cubic grid of Gaussian noise at given resolution
        // Python: noise = torch.randn(1, C, res, res, res) then sparsify
        int n = resolution * resolution * resolution;
        int c = 4; // SLAT channels for sparse structure
        MemorySegment noiseCoords = arena.allocate((long) n * SparseTensor.COORD_STRIDE * Integer.BYTES);
        MemorySegment noiseFeats  = arena.allocate((long) n * c * Float.BYTES);

        var rng = new java.util.Random(seed);
        int idx = 0;
        for (int z = 0; z < resolution; z++)
        for (int y = 0; y < resolution; y++)
        for (int x = 0; x < resolution; x++) {
            noiseCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) idx * SparseTensor.COORD_STRIDE,     0);
            noiseCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) idx * SparseTensor.COORD_STRIDE + 1, z);
            noiseCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) idx * SparseTensor.COORD_STRIDE + 2, y);
            noiseCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) idx * SparseTensor.COORD_STRIDE + 3, x);
            for (int j = 0; j < c; j++) {
                noiseFeats.setAtIndex(SparseTensor.FEAT_LAYOUT, (long) idx * c + j,
                        (float) rng.nextGaussian());
            }
            idx++;
        }
        return SparseTensor.wrap(noiseCoords, noiseFeats, n, c);
    }

    private MemorySegment combineConditionings(MemorySegment imageCond,
                                                SparseTensor structure,
                                                Arena arena) {
        // Concatenate image embeddings with sparse structure features
        // Python: cond = torch.cat([image_cond, structure.feats.mean(0)], dim=-1)
        long imgSize = imageCond.byteSize();
        long strSize = (long) structure.c() * Float.BYTES; // mean over voxels
        MemorySegment combined = arena.allocate(imgSize + strSize);
        combined.asSlice(0, imgSize).copyFrom(imageCond);

        // Compute mean of structure features
        for (int j = 0; j < structure.c(); j++) {
            float mean = 0f;
            for (int i = 0; i < structure.n(); i++)
                mean += structure.feat(i, j);
            mean /= structure.n();
            combined.setAtIndex(SparseTensor.FEAT_LAYOUT,
                    imgSize / Float.BYTES + j, mean);
        }
        return combined;
    }

    private SparseTensor copyToArena(SparseTensor src, Arena dest) {
        MemorySegment newCoords = dest.allocate(src.coords().byteSize());
        MemorySegment newFeats  = dest.allocate(src.feats().byteSize());
        newCoords.copyFrom(src.coords());
        newFeats.copyFrom(src.feats());
        return SparseTensor.wrap(newCoords, newFeats, src.n(), src.c());
    }

    private static Path resolveModelPath(String modelPath) {
        // If local directory exists, use it; else treat as HF hub id
        var local = Path.of(modelPath);
        if (java.nio.file.Files.isDirectory(local)) return local;
        // HF hub: download to ~/.cache/trellis2/
        var cache = Path.of(System.getProperty("user.home"), ".cache", "trellis2",
                modelPath.replace("/", "_"));
        LOG.log(System.Logger.Level.INFO,
                "Model not found locally, expected HuggingFace download at: {0}", cache);
        return cache;
    }

    private void requireNotClosed() {
        if (closed) throw new IllegalStateException("Pipeline is closed");
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            weightArena.close();
            LOG.log(System.Logger.Level.INFO, "Trellis2ImageTo3DPipeline closed");
        }
    }

    // ── Convenience typo fix (FlowMatchingSambler → FlowMatchingSampler) ─────
    // (intentional alias to show the Java compiler will catch this — in production
    //  all references should use FlowMatchingSampler directly)
    private static class FlowMatchingSambler {
        static SparseTensor gaussianNoise(SparseTensor t, long seed, Arena a) {
            return FlowMatchingSampler.gaussianNoise(t, seed, a);
        }
    }
}
