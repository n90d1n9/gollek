package tech.kayys.gollek.quantizer.awq;

import tech.kayys.gollek.quantizer.awq.AWQConfig;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AWQ Dequantization Engine — JDK 25 Vector API (SIMD).
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │ AWQ vs GPTQ Dequant — Key Differences │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │ GPTQ: w[i,j] = (q[i,j] − zero[g_OUT, j]) × scale[g_OUT, j] │
 * │ groups along OUTPUT dimension │
 * │ │
 * │ AWQ: w[i,j] = (q[i,j] − zero[g_IN, j]) × scale[g_IN, j] │
 * │ groups along INPUT dimension │
 * │ g_IN = i / groupSize │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * AWQ INT4 Packing (AutoAWQ GEMM format):
 * ──────────────────────────────────────────────────────────────────────
 * qweight shape: [inF / 8, outF] (8 = packFactor for 4-bit)
 * Each INT32 at qweight[pi, j] packs 8 input-feature weights
 * for the SAME output feature j, consecutive input features
 * i = pi*8 ... pi*8+7:
 *
 * qweight[pi, j] = w[pi*8+0, j] | (w[pi*8+1, j] << 4)
 * | (w[pi*8+2, j] << 8) | ...
 * | (w[pi*8+7, j] << 28)
 *
 * AWQ Zero-Point Packing:
 * ──────────────────────────────────────────────────────────────────────
 * qzeros shape: [numGroups, outF / 8]
 * Each INT32 packs 8 consecutive output-feature zero-points
 * for the same group:
 *
 * qzeros[g, pj] = zero[g, pj*8+0] | (zero[g, pj*8+1] << 4) | ...
 *
 * GEMV Packing (transposed from GEMM):
 * ──────────────────────────────────────────────────────────────────────
 * qweight shape: [inF / 8, outF] but internal bit order reverses
 * the input/output axis for the packing. The unpack logic
 * reads bits in the opposite direction.
 *
 * Vector Strategy:
 * ──────────────────────────────────────────────────────────────────────
 * - Outer loop: packed input rows (pi = i/8, stride = packFactor)
 * - Middle loop: output features in SIMD chunks of F_LANES
 * - Inner unrolled: 8 sub-iterations (one per bit-slice of the INT32)
 * - Per sub-iter: FloatVector subtract(zero) then mul(scale)
 * - This maps to ~F_LANES fused multiply-add operations per cycle
 */
public class AWQDequantizer {

    private static final Logger log = LoggerFactory.getLogger(AWQDequantizer.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    private final AWQConfig config;
    private final int packFactor;
    private final int quantMask;

    public AWQDequantizer(AWQConfig config) {
        this.config = config;
        this.packFactor = config.packFactor(); // 8 for 4-bit
        this.quantMask = config.quantMask(); // 0xF for 4-bit

        log.info("AWQDequantizer: bits={}, pack={}, mask=0x{}, format={}, SIMD={}×FP32 ({})",
                config.bits(), packFactor, Integer.toHexString(quantMask),
                config.kernelFormat(), F_LANES, FLOAT_SPECIES.toString());
    }

    // ── Main Entry Point ──────────────────────────────────────────────────────

    /**
     * Dequantizes a full AWQ layer weight matrix to FP32.
     *
     * @param qweightInts  packed INT32 weights [inF/pack, outF]
     * @param qzerosInts   packed INT32 zeros [numGroups, outF/pack] or null
     * @param scalesShorts FP16 scales (raw bits) [numGroups, outF]
     * @param inFeatures   K dimension (input channels)
     * @param outFeatures  N dimension (output channels)
     * @param output       pre-allocated FP32 output [inF × outF], row-major
     */
    public void dequantize(
            int[] qweightInts,
            int[] qzerosInts, // nullable → use implicit zero
            short[] scalesShorts,
            int inFeatures,
            int outFeatures,
            float[] output) {
        // Convert FP16 scales → FP32 once upfront
        float[] scales = fp16ToFloat32Bulk(scalesShorts);

        // Unpack zero-points → FP32 per (group, outF)
        float[] zeros = (qzerosInts != null)
                ? unpackZeros(qzerosInts, inFeatures, outFeatures)
                : buildImplicitZeros(inFeatures, outFeatures);

        switch (config.kernelFormat()) {
            case GEMM -> dequantizeGemm(qweightInts, zeros, scales, inFeatures, outFeatures, output);
            case GEMV -> dequantizeGemv(qweightInts, zeros, scales, inFeatures, outFeatures, output);
            case MARLIN -> dequantizeMarlin(qweightInts, zeros, scales, inFeatures, outFeatures, output);
        }
    }

    // ── GEMM Format (AutoAWQ default) ─────────────────────────────────────────

    /**
     * AWQ GEMM dequantization with Vector API inner loop.
     *
     * Layout: qweight[pi, j] where pi = i / packFactor
     * Unpack: for sub-bit b ∈ [0, packFactor):
     * q_val[pi*pack+b, j] = (qweight[pi, j] >> (b*bits)) & mask
     * Formula: w[i, j] = (q[i, j] − zero[g_i, j]) × scale[g_i, j]
     * g_i = i / groupSize
     */
    private void dequantizeGemm(
            int[] qweight, float[] zeros, float[] scales,
            int inF, int outF, float[] output) {
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        int packedRows = inF / packFactor; // number of packed input rows

        for (int pi = 0; pi < packedRows; pi++) {
            // Base input feature index for this packed row
            int iBase = pi * packFactor;

            // Process each of the 8 sub-elements packed in this INT32 row
            for (int b = 0; b < packFactor; b++) {
                int i = iBase + b;
                if (i >= inF)
                    break;

                int bitShift = b * config.bits();
                int g = i / config.groupSize();
                int groupOff = g * outF;

                // Process outF in SIMD chunks
                int j = 0;
                for (; j <= outF - F_LANES; j += F_LANES) {
                    // --- Gather quantized values for F_LANES output features ---
                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        // qweight[pi, j+lane]: packed row pi, output col j+lane
                        int packedWord = qweight[pi * outF + j + lane];
                        qVals[lane] = (float) ((packedWord >> bitShift) & quantMask);
                    }

                    // --- Load scale and zero vectors for this group slice ---
                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, scales, groupOff + j);
                    FloatVector vz = FloatVector.fromArray(FLOAT_SPECIES, zeros, groupOff + j);

                    // --- AWQ dequant: (q − zero) × scale ---
                    FloatVector result = vq.sub(vz).mul(vs);

                    // output[i, j] = result (row-major: i*outF + j)
                    result.intoArray(output, i * outF + j);
                }

                // Scalar tail
                for (; j < outF; j++) {
                    int qVal = (qweight[pi * outF + j] >> bitShift) & quantMask;
                    output[i * outF + j] = (qVal - zeros[groupOff + j]) * scales[groupOff + j];
                }
            }
        }
    }

    // ── GEMV Format ───────────────────────────────────────────────────────────

    /**
     * AWQ GEMV dequantization.
     *
     * GEMV packs 8 INPUT features per INT32 word, iterating OVER output features
     * in the outer loop. This is the transpose of GEMM packing.
     *
     * Layout: qweight[pi, j] where j = output feature, pi = i/pack
     * The bit-ordering within the INT32 is the same (b*bits shift),
     * but the iteration order differs from GEMM to exploit GEMV access.
     */
    private void dequantizeGemv(
            int[] qweight, float[] zeros, float[] scales,
            int inF, int outF, float[] output) {
        // GEMV layout is effectively the same memory layout as GEMM in AutoAWQ,
        // but optimized for single-token inference. We use the same unpacking
        // but swap the iteration order to maximise cache locality for GEMV.
        int packedRows = inF / packFactor;

        for (int j = 0; j < outF; j++) {
            for (int pi = 0; pi < packedRows; pi++) {
                int iBase = pi * packFactor;
                int packedWord = qweight[pi * outF + j];

                for (int b = 0; b < packFactor; b++) {
                    int i = iBase + b;
                    if (i >= inF)
                        break;
                    int g = i / config.groupSize();
                    int qVal = (packedWord >> (b * config.bits())) & quantMask;
                    float z = zeros[g * outF + j];
                    float s = scales[g * outF + j];
                    output[i * outF + j] = (qVal - z) * s;
                }
            }
        }
    }

    // ── Marlin (ExLlama v2 AWQ) Format ───────────────────────────────────────

    /**
     * Marlin / ExLlama v2 AWQ dequantization.
     *
     * Marlin uses a tiled interleaved layout optimised for GPU CUDA cores.
     * When running on CPU (JVM), we reorder to standard row-major before
     * applying the standard dequant formula. The tile size is 16×16 elements.
     *
     * Marlin interleave pattern (4-bit, tile 16×16):
     * Within each 16×16 tile, columns are interleaved in groups of 8
     * to match warp-level memory access on NVIDIA GPUs.
     * On CPU we de-interleave first, then dequantize normally.
     */
    private void dequantizeMarlin(
            int[] qweight, float[] zeros, float[] scales,
            int inF, int outF, float[] output) {
        log.debug("Marlin AWQ dequant: {}×{}", inF, outF);

        int tileRow = 16;
        int tileCol = 16;
        int packedRows = inF / packFactor;

        // De-interleave Marlin tile layout into standard row-major order
        int[] reordered = new int[qweight.length];
        int numTilesRow = (packedRows + tileRow - 1) / tileRow;
        int numTilesCol = (outF + tileCol - 1) / tileCol;

        for (int tr = 0; tr < numTilesRow; tr++) {
            for (int tc = 0; tc < numTilesCol; tc++) {
                // Within each tile: de-interleave 8-column interleave
                for (int r = 0; r < tileRow; r++) {
                    for (int c = 0; c < tileCol; c += 8) {
                        for (int k = 0; k < 8; k++) {
                            int srcRow = tr * tileRow + r;
                            int srcCol = tc * tileCol + c + k;
                            if (srcRow < packedRows && srcCol < outF) {
                                // Marlin interleave: groups of 8 columns are swapped
                                int interleavedCol = tc * tileCol + ((c / 8) % 2 == 0
                                        ? c + k
                                        : c - 8 + k + (tileCol / 2));
                                if (interleavedCol < outF) {
                                    reordered[srcRow * outF + srcCol] = qweight[srcRow * outF
                                            + Math.min(interleavedCol, outF - 1)];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply standard GEMM dequant on de-interleaved weights
        dequantizeGemm(reordered, zeros, scales, inF, outF, output);
    }

    // ── Vector-Accelerated Dequantized MatVec (W × X) ────────────────────────

    /**
     * Inference: output_vec = dequantized_weight × input_vec
     *
     * Combines dequantization and matrix-vector multiply in one pass
     * to avoid materialising the full FP32 weight matrix when only
     * inference (not export) is needed.
     *
     * @param qweightInts  packed INT32 [inF/pack, outF]
     * @param qzerosInts   packed INT32 zeros or null
     * @param scalesShorts FP16 scales
     * @param inputVec     FP32 input vector [inF]
     * @param outputVec    FP32 output vector [outF] (accumulated)
     * @param inF          input dimension
     * @param outF         output dimension
     */
    public void dequantMatVec(
            int[] qweightInts,
            int[] qzerosInts,
            short[] scalesShorts,
            float[] inputVec,
            float[] outputVec,
            int inF, int outF) {
        float[] scales = fp16ToFloat32Bulk(scalesShorts);
        float[] zeros = (qzerosInts != null)
                ? unpackZeros(qzerosInts, inF, outF)
                : buildImplicitZeros(inF, outF);

        int packedRows = inF / packFactor;

        // Zero output accumulator
        java.util.Arrays.fill(outputVec, 0f);

        for (int pi = 0; pi < packedRows; pi++) {
            int iBase = pi * packFactor;

            for (int b = 0; b < packFactor; b++) {
                int i = iBase + b;
                if (i >= inF)
                    break;

                int bitShift = b * config.bits();
                int g = i / config.groupSize();
                int gOff = g * outF;
                float xi = inputVec[i];

                int j = 0;
                for (; j <= outF - F_LANES; j += F_LANES) {
                    // Unpack F_LANES quantized values
                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        qVals[lane] = (float) ((qweightInts[pi * outF + j + lane] >> bitShift) & quantMask);
                    }

                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, scales, gOff + j);
                    FloatVector vz = FloatVector.fromArray(FLOAT_SPECIES, zeros, gOff + j);
                    FloatVector vxi = FloatVector.broadcast(FLOAT_SPECIES, xi);

                    // w = (q - z) * s, out += w * xi
                    FloatVector w = vq.sub(vz).mul(vs);
                    FloatVector acc = FloatVector.fromArray(FLOAT_SPECIES, outputVec, j);
                    w.fma(vxi, acc).intoArray(outputVec, j);
                }

                // Scalar tail
                for (; j < outF; j++) {
                    int qVal = (qweightInts[pi * outF + j] >> bitShift) & quantMask;
                    float w = (qVal - zeros[gOff + j]) * scales[gOff + j];
                    outputVec[j] += w * xi;
                }
            }
        }
    }

    // ── AWQ Activation Scaling Utilities ─────────────────────────────────────

    /**
     * Applies AWQ's input scale compensation to an activation vector.
     *
     * AWQ modifies weights as W' = W × diag(s)^(-1), so at inference
     * we must apply X' = X × diag(s) to the input (or equivalently,
     * absorb s into the preceding layer's output scaling).
     *
     * This method applies: x[i] *= activationScales[i]
     *
     * @param activation       FP32 input vector [inF] — modified in place
     * @param activationScales FP32 per-channel scale [inF]
     */
    public void applyActivationScaling(float[] activation, float[] activationScales) {
        int len = Math.min(activation.length, activationScales.length);
        int i = 0;

        for (; i <= len - F_LANES; i += F_LANES) {
            FloatVector va = FloatVector.fromArray(FLOAT_SPECIES, activation, i);
            FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, activationScales, i);
            va.mul(vs).intoArray(activation, i);
        }
        for (; i < len; i++)
            activation[i] *= activationScales[i];
    }

    /**
     * Fuses two consecutive linear layers when the first has an AWQ scale.
     *
     * When W1's output is scaled by s and W2's input expects x*s,
     * we can absorb the scale into W2's dequantized weights:
     * W2_eff[:, i] = W2[:, i] / s[i]
     *
     * This eliminates the runtime scaling op at a one-time cost during load.
     */
    public void absorbActivationScale(float[] w2Dequant, float[] activationScale,
            int inF, int outF) {
        for (int i = 0; i < inF; i++) {
            if (i >= activationScale.length || activationScale[i] == 0f)
                continue;
            float invScale = 1.0f / activationScale[i];
            int rowOff = i * outF;
            int j = 0;
            for (; j <= outF - F_LANES; j += F_LANES) {
                FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, w2Dequant, rowOff + j);
                FloatVector vis = FloatVector.broadcast(FLOAT_SPECIES, invScale);
                vw.mul(vis).intoArray(w2Dequant, rowOff + j);
            }
            for (; j < outF; j++)
                w2Dequant[rowOff + j] *= invScale;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Bulk FP16 → FP32 conversion array.
     */
    public float[] fp16ToFloat32Bulk(short[] fp16) {
        float[] out = new float[fp16.length];
        for (int i = 0; i < fp16.length; i++) {
            out[i] = MemoryAllocator.fp16ToFloat32(fp16[i]);
        }
        return out;
    }

    /**
     * Unpacks INT32-packed zero-points into a flat FP32 array [numGroups × outF].
     *
     * AWQ zeros packing: qzeros[g, pj] holds 8 zero-point values for
     * output features pj*8 ... pj*8+7 in group g.
     *
     * The AutoAWQ convention adds +1 to the raw zero (same as GPTQ).
     */
    public float[] unpackZeros(int[] qzerosInts, int inF, int outF) {
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        int packedCols = outF / packFactor; // number of INT32 cols in qzeros
        float[] zeros = new float[numGroups * outF];

        for (int g = 0; g < numGroups; g++) {
            for (int pj = 0; pj < packedCols; pj++) {
                if (g * packedCols + pj >= qzerosInts.length)
                    break;
                int packed = qzerosInts[g * packedCols + pj];
                for (int b = 0; b < packFactor; b++) {
                    int j = pj * packFactor + b;
                    if (j < outF) {
                        // AutoAWQ +1 convention
                        zeros[g * outF + j] = (float) (((packed >> (b * config.bits())) & quantMask) + 1);
                    }
                }
            }
        }
        return zeros;
    }

    /**
     * Builds implicit zero array for symmetric AWQ (no stored qzeros).
     * For 4-bit symmetric: zero = 2^(bits-1) = 8
     */
    public float[] buildImplicitZeros(int inF, int outF) {
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        float implicitZ = config.implicitZero();
        float[] zeros = new float[numGroups * outF];
        java.util.Arrays.fill(zeros, implicitZ);
        return zeros;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    public static void printCapabilities() {
        System.out.println("=== AWQ Vector API Capabilities ===");
        System.out.printf("Float SIMD species : %s (%d lanes, %d-bit)%n",
                FLOAT_SPECIES.toString(), FLOAT_SPECIES.length(), FLOAT_SPECIES.vectorBitSize());
        System.out.printf("Int   SIMD species : %s (%d lanes, %d-bit)%n",
                INT_SPECIES.toString(), INT_SPECIES.length(), INT_SPECIES.vectorBitSize());
        System.out.printf("Effective ISA      : %s%n",
                FLOAT_SPECIES.vectorBitSize() >= 512 ? "AVX-512 (16 FP32 lanes)"
                        : FLOAT_SPECIES.vectorBitSize() >= 256 ? "AVX2 / NEON+ (8 lanes)" : "SSE4 / NEON (4 lanes)");
    }
}
