package tech.kayys.gollek.quantizer.turboquant;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * BitsAndBytes NF4 (NormalFloat-4) and INT8 Dequantization Engine.
 *
 * BitsAndBytes uses a fundamentally different approach from GPTQ/AWQ:
 * it operates at RUNTIME, quantizing weights on-the-fly during forward pass
 * rather than pre-quantizing offline. For Java inference, we dequantize
 * the stored NF4/INT8 weights back to FP32.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ NF4 (NormalFloat-4) — the QLoRA quantization scheme │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ Key insight: LLM weights are approximately normally distributed. │
 * │ Standard INT4 wastes precision at extremes (±8 mapped to outliers). │
 * │ NF4 uses 16 quantization levels with EQUAL probability mass under │
 * │ N(0,1), maximizing information content for normal-distributed data. │
 * │ │
 * │ NF4 lookup table (16 values, information-theoretically optimal): │
 * │ {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0933, 0.0, │
 * │ 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}│
 * │ │
 * │ Tensor layout (BitsAndBytes NF4, per-block): │
 * │ weight : packed INT4 nibbles, shape [n/2] │
 * │ absmax : per-block absmax FP32, shape [n/blockSize] │
 * │ (optional): double-quant absmax FP8 + meta │
 * │ │
 * │ Dequant: w = NF4_TABLE[nibble] × absmax[block] │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Double Quantization (QLoRA):
 * The absmax values themselves are quantized (FP8) to save memory.
 * A secondary absmax (dq_absmax) dequantizes the primary absmax first.
 *
 * absmax_fp32[b] = FP8_TABLE[absmax_fp8[b]] × dq_absmax[b/dq_block]
 *
 * INT8 LLM.int8() format:
 * Uses column-wise INT8 quantization with per-column FP32 scales.
 * Outlier weights are handled separately in FP16.
 * w[i,j] = int8_weight[i,j] × col_scale[j]
 */
public class BnBDequantizer {

    private static final Logger log = LoggerFactory.getLogger(BnBDequantizer.class);

    // Use the preferred vector species for the current platform
    // In JDK 25+, PREFERRED_SPECIES is accessed via VectorSpecies.ofPreferred()
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    /**
     * The 16 NormalFloat-4 quantization levels.
     * These are the unique values representable by NF4, in ascending order.
     * Index i maps to NF4_TABLE[i].
     *
     * Derivation: inverse CDF of N(0,1) at 16 equal-probability quantiles,
     * normalized so max(|values|) = 1.0.
     */
    public static final float[] NF4_TABLE = {
            -1.0f,
            -0.6961928009986877f,
            -0.5250730514526367f,
            -0.39491748809814453f,
            -0.28444138169288635f,
            -0.18477343022823334f,
            -0.09105003625154495f,
            0.0f,
            0.07958029955625534f,
            0.16093020141124725f,
            0.24611230194568634f,
            0.33791524171829224f,
            0.44070982933044434f,
            0.5626170039176941f,
            0.7229568362236023f,
            1.0f
    };

    /**
     * FP8 lookup table for double quantization (8-bit absmax storage).
     * Maps FP8 E4M3 encoded byte to float.
     * (Simplified — full E4M3 range; BnB uses a subset)
     */
    private static final float[] FP8_TABLE;
    static {
        FP8_TABLE = new float[256];
        // E4M3: sign(1) | exponent(4) | mantissa(3)
        for (int i = 0; i < 256; i++) {
            int sign = (i >> 7) & 1;
            int exp = (i >> 3) & 0xF;
            int mant = i & 0x7;
            float v;
            if (exp == 0)
                v = (sign == 0 ? 1 : -1) * mant / 8.0f * (float) Math.pow(2, -6);
            else if (exp == 15)
                v = Float.NaN; // NaN in E4M3
            else
                v = (sign == 0 ? 1 : -1) * (1.0f + mant / 8.0f) * (float) Math.pow(2, exp - 7);
            FP8_TABLE[i] = v;
        }
    }

    private final int blockSize; // default 64 for NF4
    private final boolean doubleQuant; // whether absmax is also quantized

    public BnBDequantizer(int blockSize, boolean doubleQuant) {
        this.blockSize = blockSize;
        this.doubleQuant = doubleQuant;
        log.info("BnBDequantizer: blockSize={}, doubleQuant={}, SIMD={}×FP32",
                blockSize, doubleQuant, F_LANES);
    }

    public BnBDequantizer() {
        this(64, false);
    }

    // ── NF4 Dequantization ────────────────────────────────────────────────────

    /**
     * Dequantizes NF4-packed weights to FP32.
     *
     * @param packedNibbles INT4 weights packed as bytes (2 per byte, lo nibble
     *                      first)
     *                      shape: [numElements / 2]
     * @param absmax        per-block scale factors (FP32), shape: [numElements /
     *                      blockSize]
     * @param numElements   total number of original weight elements
     * @param output        pre-allocated FP32 output [numElements]
     */
    public void dequantNF4(byte[] packedNibbles, float[] absmax, int numElements, float[] output) {
        int numBlocks = (numElements + blockSize - 1) / blockSize;

        for (int b = 0; b < numBlocks; b++) {
            float scale = absmax[b];
            int elemBase = b * blockSize;
            int elemEnd = Math.min(elemBase + blockSize, numElements);
            int count = elemEnd - elemBase;

            // Process SIMD-friendly pairs of elements (each byte holds 2 nibbles)
            int i = 0;

            // SIMD: gather F_LANES decoded NF4 values and scale together
            for (; i <= count - F_LANES; i += F_LANES) {
                float[] decoded = new float[F_LANES];
                for (int lane = 0; lane < F_LANES; lane++) {
                    int elem = elemBase + i + lane;
                    int byteIdx = elem / 2;
                    int nibble = (elem % 2 == 0)
                            ? (packedNibbles[byteIdx] & 0xF)
                            : ((packedNibbles[byteIdx] >> 4) & 0xF);
                    decoded[lane] = NF4_TABLE[nibble];
                }
                FloatVector vd = FloatVector.fromArray(FLOAT_SPECIES, decoded, 0);
                FloatVector vs = FloatVector.broadcast(FLOAT_SPECIES, scale);
                vd.mul(vs).intoArray(output, elemBase + i);
            }

            // Scalar tail
            for (; i < count; i++) {
                int elem = elemBase + i;
                int byteIdx = elem / 2;
                int nibble = (elem % 2 == 0)
                        ? (packedNibbles[byteIdx] & 0xF)
                        : ((packedNibbles[byteIdx] >> 4) & 0xF);
                output[elem] = NF4_TABLE[nibble] * scale;
            }
        }
    }

    /**
     * Dequantizes NF4 with double quantization (QLoRA style).
     *
     * First dequantizes the FP8-encoded absmax values, then uses those
     * to dequantize the NF4 weights.
     *
     * @param packedNibbles  NF4 packed weights
     * @param absmax_fp8     per-block absmax encoded as FP8 bytes
     * @param dq_absmax_fp32 per-double-block absmax (FP32), secondary scale
     * @param dqBlockSize    double-quant block size (typically 256)
     * @param numElements    total elements
     * @param output         FP32 output
     */
    public void dequantNF4DoubleQuant(
            byte[] packedNibbles,
            byte[] absmax_fp8,
            float[] dq_absmax_fp32,
            int dqBlockSize,
            int numElements,
            float[] output) {
        int numBlocks = (numElements + blockSize - 1) / blockSize;
        float[] absmax = new float[numBlocks];

        // Step 1: dequantize absmax FP8 → FP32 using secondary scale
        for (int b = 0; b < numBlocks; b++) {
            int dqBlock = b / dqBlockSize;
            float dqScale = dqBlock < dq_absmax_fp32.length ? dq_absmax_fp32[dqBlock] : 1.0f;
            absmax[b] = FP8_TABLE[absmax_fp8[b] & 0xFF] * dqScale;
        }

        // Step 2: dequantize NF4 with recovered absmax
        dequantNF4(packedNibbles, absmax, numElements, output);
    }

    // ── INT8 (LLM.int8()) Dequantization ──────────────────────────────────────

    /**
     * Dequantizes LLM.int8() column-wise INT8 weights to FP32.
     *
     * LLM.int8() uses threshold-based quantization:
     * - Normal weights: column-wise INT8 with per-column FP32 scale
     * - Outlier weights (|x| > threshold): stored in FP16 and added back
     *
     * @param int8Weights INT8 weight matrix [outF, inF], row-major
     * @param colScales   per-column FP32 scale [inF]
     * @param outF        output features (rows)
     * @param inF         input features (cols)
     * @param output      FP32 output [outF, inF]
     */
    public void dequantInt8(byte[] int8Weights, float[] colScales,
            int outF, int inF, float[] output) {
        for (int j = 0; j < outF; j++) {
            int rowOff = j * inF;
            int i = 0;

            // SIMD: process F_LANES input columns at once
            for (; i <= inF - F_LANES; i += F_LANES) {
                float[] vals = new float[F_LANES];
                for (int lane = 0; lane < F_LANES; lane++) {
                    vals[lane] = (float) int8Weights[rowOff + i + lane];
                }
                FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, vals, 0);
                FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, colScales, i);
                vq.mul(vs).intoArray(output, rowOff + i);
            }

            // Scalar tail
            for (; i < inF; i++) {
                output[rowOff + i] = int8Weights[rowOff + i] * colScales[i];
            }
        }
    }

    /**
     * Adds outlier FP16 contributions back to an INT8-dequantized matrix.
     *
     * LLM.int8() stores large-magnitude weights separately in FP16 to
     * avoid quantization error for outlier channels. This method re-adds them.
     *
     * @param output      INT8-dequantized weights (modified in-place)
     * @param outlierRows row indices of outlier elements
     * @param outlierCols col indices of outlier elements
     * @param outlierVals FP16 outlier values (as raw shorts)
     * @param inF         matrix column count
     */
    public void addOutliers(float[] output, int[] outlierRows, int[] outlierCols,
            short[] outlierVals, int inF) {
        for (int k = 0; k < outlierRows.length; k++) {
            int idx = outlierRows[k] * inF + outlierCols[k];
            float val = fp16ToFloat32(outlierVals[k]);
            output[idx] += val;
        }
    }

    // ── FP4 (FP4 quantization, used in some BnB variants) ────────────────────

    /**
     * Dequantizes FP4 (4-bit floating point) weights.
     *
     * FP4 uses a 4-bit floating point representation: 1 sign + 2 exp + 1 mantissa.
     * The 16 FP4 values map to the following floats:
     * 0b0000=0.0, 0b0001=0.0625, 0b0010=0.125, ... 0b1111=-6.0
     */
    private static final float[] FP4_TABLE = {
            0.0f, 0.0625f, 0.125f, 0.25f,
            0.375f, 0.5f, 0.75f, 1.0f,
            -0.0f, -0.0625f, -0.125f, -0.25f,
            -0.375f, -0.5f, -0.75f, -1.0f
    };

    public void dequantFP4(byte[] packedNibbles, float[] absmax, int numElements, float[] output) {
        for (int i = 0; i < numElements; i++) {
            int b = i / blockSize;
            int byteIdx = i / 2;
            int nibble = (i % 2 == 0)
                    ? (packedNibbles[byteIdx] & 0xF)
                    : ((packedNibbles[byteIdx] >> 4) & 0xF);
            output[i] = FP4_TABLE[nibble] * absmax[b];
        }
    }

    // ── Quantize Utility (for testing) ────────────────────────────────────────

    /**
     * Quantizes FP32 weights to NF4 packed format.
     * Useful for unit testing the round-trip.
     *
     * @param weights   FP32 weights
     * @param packedOut output packed nibbles [numElements / 2]
     * @param absmaxOut output per-block absmax [numElements / blockSize]
     */
    public void quantizeNF4(float[] weights, byte[] packedOut, float[] absmaxOut) {
        int numBlocks = (weights.length + blockSize - 1) / blockSize;

        for (int b = 0; b < numBlocks; b++) {
            int base = b * blockSize;
            int end = Math.min(base + blockSize, weights.length);

            // Find absmax for this block
            float amax = 0f;
            for (int i = base; i < end; i++)
                amax = Math.max(amax, Math.abs(weights[i]));
            absmaxOut[b] = amax;
            float scale = amax > 0 ? 1.0f / amax : 1.0f;

            // Quantize each element to nearest NF4 value
            for (int i = base; i < end; i++) {
                float normalized = weights[i] * scale;
                int nibble = nearestNF4(normalized);
                int byteIdx = i / 2;
                if (i % 2 == 0) {
                    packedOut[byteIdx] = (byte) (nibble & 0xF);
                } else {
                    packedOut[byteIdx] |= (byte) ((nibble & 0xF) << 4);
                }
            }
        }
    }

    /** Finds the nearest NF4 table index for a normalized value ∈ [-1, 1]. */
    private static int nearestNF4(float v) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < NF4_TABLE.length; i++) {
            float d = Math.abs(v - NF4_TABLE[i]);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        return best;
    }

    // ── FP16 utility ──────────────────────────────────────────────────────────

    private static float fp16ToFloat32(short s) {
        int h = s & 0xFFFF;
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
            return 0f;
        if (exp == 31)
            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    public static void printCapabilities() {
        System.out.println("=== BitsAndBytes Dequantizer ===");
        System.out.printf("Formats: NF4, NF4+DoubleQuant, FP4, INT8 (LLM.int8())%n");
        System.out.printf("NF4 table size: %d levels%n", NF4_TABLE.length);
        System.out.printf("SIMD: %s (%d float lanes)%n",
                FLOAT_SPECIES.toString(), FLOAT_SPECIES.length());
    }
}
