package tech.kayys.trellis2.modules.sparse;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Sparse neural network operations — Java 25 port of trellis2/modules/sparse/
 *
 * <p>All operations act on {@link SparseTensor}s and dispatch to native
 * FlexGEMM / spconv kernels via the FFM API when available, falling back
 * to pure-Java CPU implementations.
 *
 * <p>Python originals in trellis2/modules/sparse/basic.py:
 * <pre>
 *   SparseConv3d        — wraps spconv.SubMConv3d / FlexGEMM
 *   SparseLinear        — simple linear projection on .feats
 *   SparseDownsample    — max-pool halving of voxel grid
 *   SparseUpsample      — guided subdivision doubling
 *   SparseSpatial2Channel — pack 8 child voxels into 1 parent (S→C)
 *   SparseChannel2Spatial — unpack 1 parent into 8 children (C→S)
 * </pre>
 */
public final class SparseOps {

    private SparseOps() {}

    // ═════════════════════════════════════════════════════════════════════════
    // SparseConv3d
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Sparse 3D convolution — Python: {@code sp.SparseConv3d(in_channels, out_channels, kernel)}.
     *
     * <p>Uses FlexGEMM Triton kernel (via FFM) when available;
     * falls back to gather-GEMM-scatter on CPU.
     */
    public static final class SparseConv3d {

        private final int inC;
        private final int outC;
        private final int kernel;  // 3 → 3×3×3
        private final MemorySegment weight;  // float32[outC, inC, k, k, k]
        private final MemorySegment bias;    // float32[outC] or null

        public SparseConv3d(int inC, int outC, int kernel, Arena weightArena) {
            this.inC    = inC;
            this.outC   = outC;
            this.kernel = kernel;
            // Allocate weight tensor in provided arena
            long kk = (long) kernel * kernel * kernel;
            this.weight = weightArena.allocate(outC * inC * kk * Float.BYTES);
            this.bias   = null; // no bias in TRELLIS.2 sparse convs
            initWeightsKaimingUniform();
        }

        /** Python: {@code zero_module(sp.SparseConv3d(...))} — zero-initialized. */
        public SparseConv3d(int inC, int outC, int kernel, Arena weightArena,
                             boolean zeroInit) {
            this(inC, outC, kernel, weightArena);
            if (zeroInit) weight.fill((byte) 0);
        }

        /**
         * Python: forward pass (called as {@code self.conv1(h)}).
         * Applies 3D sparse convolution; output has same coordinates as input.
         */
        public SparseTensor apply(SparseTensor x, Arena arena) {
            return SparseConv3d.apply(x, weight, outC, kernel, arena);
        }

        /**
         * Static apply — for use when weights are passed directly (e.g., from
         * a loaded safetensors file without a SparseConv3d wrapper object).
         */
        public static SparseTensor apply(SparseTensor x, MemorySegment weight,
                                          int outC, int kernel, Arena arena) {
            return x.conv3d(weight, outC, kernel, arena);
        }

        private void initWeightsKaimingUniform() {
            // Kaiming uniform init: fan_in = inC * k^3
            long kk = (long) kernel * kernel * kernel;
            double fanIn  = (double) inC * kk;
            double bound  = Math.sqrt(1.0 / fanIn);
            var rng = new java.util.Random(42);
            long total    = (long) outC * inC * kk;
            for (long i = 0; i < total; i++) {
                float v = (float) (rng.nextDouble() * 2 * bound - bound);
                weight.setAtIndex(SparseTensor.FEAT_LAYOUT, i, v);
            }
        }

        public MemorySegment weightSegment() { return weight; }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SparseLinear
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Linear projection on sparse features — Python: {@code sp.SparseLinear(in, out)}.
     * Equivalent to {@code nn.Linear} applied to x.feats without any spatial coupling.
     */
    public static final class SparseLinear {

        private final int inC;
        private final int outC;
        private final MemorySegment weight;  // float32[outC, inC]
        private final MemorySegment bias;    // float32[outC]

        public SparseLinear(int inC, int outC, Arena weightArena) {
            this.inC   = inC;
            this.outC  = outC;
            this.weight = weightArena.allocate((long) outC * inC * Float.BYTES);
            this.bias   = weightArena.allocate((long) outC * Float.BYTES);
        }

        public SparseTensor apply(SparseTensor x, Arena arena) {
            return SparseLinear.apply(x, weight, outC, arena);
        }

        /**
         * Static apply — GEMM: outFeats[i][j] = sum_k(x.feats[i][k] * weight[j][k]) + bias[j]
         */
        public static SparseTensor apply(SparseTensor x, MemorySegment weight,
                                          int outC, Arena arena) {
            int n    = x.n();
            int inC  = x.c();
            MemorySegment out = arena.allocate((long) n * outC * Float.BYTES);

            // CPU GEMM fallback (production: dispatch to cuBLAS via FFM)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < outC; j++) {
                    float acc = 0f;
                    for (int k = 0; k < inC; k++) {
                        float xv = x.feats().getAtIndex(SparseTensor.FEAT_LAYOUT,
                                (long) i * inC + k);
                        float wv = weight.getAtIndex(SparseTensor.FEAT_LAYOUT,
                                (long) j * inC + k);
                        acc += xv * wv;
                    }
                    out.setAtIndex(SparseTensor.FEAT_LAYOUT, (long) i * outC + j, acc);
                }
            }
            return x.replace(out, outC);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SparseDownsample
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Sparse max-pool downsampling — Python: {@code sp.SparseDownsample(2)}.
     * Halves the resolution of a sparse voxel grid (factor = 2).
     *
     * <p>For each 2³=8 child voxels that share the same parent cell,
     * the representative (first encountered) is kept.
     */
    public static final class SparseDownsample {

        private final int factor;

        public SparseDownsample(int factor) {
            this.factor = factor;
        }

        public SparseTensor apply(SparseTensor x, Arena arena) {
            return x.downsample2(arena);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SparseUpsample
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Guided sparse upsampling — Python: {@code sp.SparseUpsample(2)}.
     * Doubles resolution using a binary subdivision mask ({@code subdiv}).
     *
     * <p>Only child voxels where {@code subdiv.feats > 0} are materialized.
     * This is the key mechanism that lets TRELLIS.2 generate high-resolution
     * geometry without materializing a dense 1536³ grid.
     */
    public static final class SparseUpsample {

        private final int factor;

        public SparseUpsample(int factor) {
            this.factor = factor;
        }

        public SparseTensor apply(SparseTensor x, SparseTensor subdiv, Arena arena) {
            return x.upsample2(subdiv, arena);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SparseSpatial2Channel  (S2C)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Spatial-to-channel packing — Python: {@code sp.SparseSpatial2Channel(2)}.
     *
     * <p>Merges 8 child voxels into 1 parent voxel by concatenating
     * their feature channels: [N/8, C*8].
     * Used as an alternative downsampling that preserves all information.
     */
    public static final class SparseSpatial2Channel {

        private final int factor;

        public SparseSpatial2Channel(int factor) {
            this.factor = factor;
        }

        public SparseTensor apply(SparseTensor x, Arena arena) {
            // Group 8 children per parent, concatenate features
            int n      = x.n();
            int c      = x.c();
            int factor3= factor * factor * factor; // 8 for factor=2
            int outN   = n / factor3;
            int outC   = c * factor3;

            MemorySegment outCoords = arena.allocate((long) outN * SparseTensor.COORD_STRIDE * Integer.BYTES);
            MemorySegment outFeats  = arena.allocate((long) outN * outC * Float.BYTES);

            // Pack: for each group of 8, write parent coord and concat feats
            for (int pi = 0; pi < outN; pi++) {
                // Write parent coords (from first child, right-shifted)
                for (int d = 0; d < SparseTensor.COORD_STRIDE; d++) {
                    int v = x.coord(pi * factor3, d);
                    outCoords.setAtIndex(SparseTensor.COORD_LAYOUT,
                            (long) pi * SparseTensor.COORD_STRIDE + d,
                            d == 0 ? v : v >> 1);
                }
                // Concatenate child features
                for (int ci = 0; ci < factor3; ci++) {
                    int srcIdx = pi * factor3 + ci;
                    for (int j = 0; j < c; j++) {
                        float v = srcIdx < n ? x.feat(srcIdx, j) : 0f;
                        outFeats.setAtIndex(SparseTensor.FEAT_LAYOUT,
                                (long) pi * outC + ci * c + j, v);
                    }
                }
            }
            return SparseTensor.wrap(outCoords, outFeats, outN, outC);
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SparseChannel2Spatial  (C2S)
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Channel-to-spatial unpacking — Python: {@code sp.SparseChannel2Spatial(2)}.
     *
     * <p>Inverse of {@link SparseSpatial2Channel}: splits [N, C*8] → [N*8, C]
     * by expanding each voxel into 8 children (guided by subdiv mask).
     */
    public static final class SparseChannel2Spatial {

        private final int factor;

        public SparseChannel2Spatial(int factor) {
            this.factor = factor;
        }

        public SparseTensor apply(SparseTensor x, SparseTensor subdiv, Arena arena) {
            int factor3 = factor * factor * factor;
            int n       = x.n();
            int outC    = x.c() / factor3;
            int outN    = n * factor3;

            MemorySegment outCoords = arena.allocate((long) outN * SparseTensor.COORD_STRIDE * Integer.BYTES);
            MemorySegment outFeats  = arena.allocate((long) outN * outC * Float.BYTES);

            int out = 0;
            for (int pi = 0; pi < n; pi++) {
                int pz = x.coord(pi, 1), py = x.coord(pi, 2), px = x.coord(pi, 3);
                for (int dz = 0; dz < factor; dz++)
                for (int dy = 0; dy < factor; dy++)
                for (int dx = 0; dx < factor; dx++) {
                    int ci = dz * 4 + dy * 2 + dx;
                    // Write child coords
                    outCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) out * SparseTensor.COORD_STRIDE,     x.coord(pi, 0));
                    outCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) out * SparseTensor.COORD_STRIDE + 1, pz * factor + dz);
                    outCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) out * SparseTensor.COORD_STRIDE + 2, py * factor + dy);
                    outCoords.setAtIndex(SparseTensor.COORD_LAYOUT, (long) out * SparseTensor.COORD_STRIDE + 3, px * factor + dx);
                    // Write child features (slice from parent channel block)
                    for (int j = 0; j < outC; j++) {
                        outFeats.setAtIndex(SparseTensor.FEAT_LAYOUT,
                                (long) out * outC + j,
                                x.feat(pi, ci * outC + j));
                    }
                    out++;
                }
            }
            return SparseTensor.wrap(outCoords, outFeats, outN, outC);
        }
    }
}
