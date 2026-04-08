package tech.kayys.gollek.quantizer.turboquant;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SqueezeLLM Dequantization Engine — JDK 25 Vector API.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ SqueezeLLM Algorithm │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ SqueezeLLM (Kim et al., 2023) observes that LLM weight distributions │
 * │ are heavy-tailed: most weights can be aggressively quantized, but │
 * │ a small fraction of high-sensitivity (salient) weights must be kept │
 * │ in full precision. │
 * │ │
 * │ Two-component storage: │
 * │ 1. DENSE: The bulk of weights stored in INT4/INT8 with a per-column │
 * │ non-uniform quantization codebook (lookup table, not uniform INT4)│
 * │ 2. SPARSE: Salient weights stored as sparse matrix in CSR format │
 * │ using FP16 (or FP32) with their exact original values │
 * │ │
 * │ Key advantages: │
 * │ - Non-uniform codebook: 16 INT4 values are chosen to minimise MSE │
 * │ for each column's distribution (not fixed uniform grid) │
 * │ - Sparse outliers: ~0.5% of weights captured at full precision │
 * │ - Combined: better quality than uniform INT4 at same bit rate │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * SqueezeLLM Tensor Layout:
 * ──────────────────────────────────────────────────────────────────────────
 * layer.weight : INT32 packed INT4 [outF/pack, inF]
 * layer.lookup : FP32 codebook [inF, 2^bits] (per-column LUT)
 * or FP16 [inF, 16] for 4-bit
 * layer.weight_nnz : FP16/FP32 sparse values [nnz] (non-zero outliers)
 * layer.dense_idx : INT32 CSR column indices [nnz]
 * layer.dense_ptr : INT32 CSR row pointers [outF + 1]
 *
 * Dequantization:
 * w_dense[i,j] = lookup[j, qweight[i,j]] ← non-uniform dequant
 * w_total[i,j] = w_dense[i,j] + sparse[i,j] ← add outliers back
 *
 * where sparse[i,j] is 0 for most (i,j) pairs, and non-zero (stored in CSR)
 * for the ~0.5% salient positions.
 *
 * CSR (Compressed Sparse Row) format:
 * dense_ptr[i] = start index in weight_nnz/dense_idx for row i
 * dense_ptr[i+1] = end index (exclusive)
 * For row i: sparse non-zeros are at cols
 * dense_idx[dense_ptr[i]..dense_ptr[i+1]]
 * with values weight_nnz[dense_ptr[i]..dense_ptr[i+1]]
 */
public class SqueezeLLMDequantizer {

    private static final Logger log = LoggerFactory.getLogger(SqueezeLLMDequantizer.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    private final int bits;
    private final int packFactor;
    private final int quantMask;

    public SqueezeLLMDequantizer() {
        this(4);
    }

    public SqueezeLLMDequantizer(int bits) {
        this.bits = bits;
        this.packFactor = 32 / bits;
        this.quantMask = (1 << bits) - 1;
        log.info("SqueezeLLMDequantizer: bits={}, packFactor={}, SIMD={}×FP32",
                bits, packFactor, F_LANES);
    }

    // ── Dense (Non-Uniform Lookup) Dequantization ─────────────────────────────

    /**
     * Dequantizes the dense component using per-column lookup tables.
     *
     * For each weight w[i,j]:
     * nibble = extract bits from qweight INT32
     * w[i,j] = lookup[j, nibble] (column j's private codebook)
     *
     * The lookup table has shape [inF, 2^bits] — each input column j
     * has its own set of 16 (for 4-bit) quantization levels optimized
     * for that column's weight distribution.
     *
     * @param qweightInts packed INT32 [outF/pack, inF]
     * @param lookupFp32  per-column codebook [inF × 2^bits] row-major
     *                    i.e., lookup[col=j, level=k] at index j*(2^bits)+k
     * @param inF         input features
     * @param outF        output features
     * @param output      FP32 dense weights [outF × inF]
     */
    public void dequantizeDense(
            int[] qweightInts,
            float[] lookupFp32,
            int inF,
            int outF,
            float[] output) {
        int numLevels = 1 << bits; // 16 for 4-bit
        int packedRows = outF / packFactor;

        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;
                int bitShift = b * bits;
                int outRow = j * inF;

                // For each input column, look up the non-uniform dequant value
                int i = 0;
                // SIMD: gather F_LANES lookups per cycle
                for (; i <= inF - F_LANES; i += F_LANES) {
                    float[] w = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        int col = i + lane;
                        int nibble = (qweightInts[pr * inF + col] >> bitShift) & quantMask;
                        // Lookup: column col's codebook entry nibble
                        w[lane] = lookupFp32[col * numLevels + nibble];
                    }
                    FloatVector.fromArray(FLOAT_SPECIES, w, 0).intoArray(output, outRow + i);
                }
                // Scalar tail
                for (; i < inF; i++) {
                    int nibble = (qweightInts[pr * inF + i] >> bitShift) & quantMask;
                    output[outRow + i] = lookupFp32[i * numLevels + nibble];
                }
            }
        }
    }

    // ── Sparse Scatter-Add (outlier reconstruction) ───────────────────────────

    /**
     * Adds the sparse outlier component back to the dense dequantized weight.
     *
     * Uses CSR (Compressed Sparse Row) format. For each row i (output feature):
     * for k in [dense_ptr[i], dense_ptr[i+1]):
     * output[i, dense_idx[k]] += weight_nnz[k]
     *
     * @param output    dense dequantized weights [outF × inF] (modified in-place)
     * @param weightNnz FP32 sparse outlier values [nnz]
     * @param denseIdx  INT32 column indices for each non-zero [nnz]
     * @param densePtr  INT32 CSR row pointers [outF + 1]
     * @param outF      output features (rows)
     * @param inF       input features (cols, for bounds checking)
     */
    public void scatterAddSparse(
            float[] output,
            float[] weightNnz,
            int[] denseIdx,
            int[] densePtr,
            int outF,
            int inF) {
        for (int i = 0; i < outF; i++) {
            int rowBase = i * inF;
            int start = densePtr[i];
            int end = densePtr[i + 1];

            for (int k = start; k < end; k++) {
                int col = denseIdx[k];
                float val = weightNnz[k];
                if (col >= 0 && col < inF) {
                    output[rowBase + col] += val;
                }
            }
        }
    }

    /**
     * Converts FP16 sparse values to FP32 scatter-add.
     * Used when weight_nnz is stored as FP16 (common in SqueezeLLM exports).
     */
    public void scatterAddSparseFp16(
            float[] output,
            short[] weightNnzFp16,
            int[] denseIdx,
            int[] densePtr,
            int outF,
            int inF) {
        for (int i = 0; i < outF; i++) {
            int rowBase = i * inF;
            int start = densePtr[i];
            int end = densePtr[i + 1];
            for (int k = start; k < end; k++) {
                int col = denseIdx[k];
                float val = fp16ToFloat32(weightNnzFp16[k]);
                if (col >= 0 && col < inF)
                    output[rowBase + col] += val;
            }
        }
    }

    // ── Full Dequant (Dense + Sparse) ─────────────────────────────────────────

    /**
     * Full SqueezeLLM dequantization: dense lookup + sparse scatter.
     *
     * @param qweightInts packed INT32 [outF/pack, inF]
     * @param lookupFp32  codebook [inF × numLevels]
     * @param weightNnz   FP32 sparse values
     * @param denseIdx    CSR column indices
     * @param densePtr    CSR row pointers [outF+1]
     * @param inF,        outF dimensions
     * @param output      output [outF × inF]
     */
    public void dequantize(
            int[] qweightInts,
            float[] lookupFp32,
            float[] weightNnz,
            int[] denseIdx,
            int[] densePtr,
            int inF,
            int outF,
            float[] output) {
        // Step 1: dense non-uniform lookup
        dequantizeDense(qweightInts, lookupFp32, inF, outF, output);

        // Step 2: sparse scatter-add for outliers
        if (weightNnz != null && denseIdx != null && densePtr != null) {
            scatterAddSparse(output, weightNnz, denseIdx, densePtr, outF, inF);
            log.debug("Scatter-add {} sparse non-zeros", weightNnz.length);
        }
    }

    // ── Fused Lookup MatVec ───────────────────────────────────────────────────

    /**
     * Fused SqueezeLLM inference: outputVec = W_dequant × inputVec
     *
     * Combines lookup-based dequant and dot product without materializing W.
     * Dense component: lookup-dequant each weight on-the-fly during dot product.
     * Sparse component: scatter-multiply the sparse values.
     *
     * @param qweightInts [outF/pack, inF]
     * @param lookupFp32  [inF × numLevels]
     * @param weightNnz   FP32 sparse values
     * @param denseIdx    CSR column indices
     * @param densePtr    CSR row pointers
     * @param inputVec    FP32 input [inF]
     * @param outputVec   FP32 output [outF] — overwritten
     */
    public void dequantMatVec(
            int[] qweightInts,
            float[] lookupFp32,
            float[] weightNnz,
            int[] denseIdx,
            int[] densePtr,
            float[] inputVec,
            float[] outputVec,
            int inF, int outF) {
        int numLevels = 1 << bits;
        int packedRows = outF / packFactor;
        java.util.Arrays.fill(outputVec, 0f);

        // Dense: fused lookup + dot
        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;
                int bitShift = b * bits;
                FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
                int i = 0;

                for (; i <= inF - F_LANES; i += F_LANES) {
                    float[] w = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        int col = i + lane;
                        int nibble = (qweightInts[pr * inF + col] >> bitShift) & quantMask;
                        w[lane] = lookupFp32[col * numLevels + nibble];
                    }
                    FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, w, 0);
                    FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, inputVec, i);
                    acc = vw.fma(vx, acc);
                }
                float dot = acc.reduceLanes(VectorOperators.ADD);
                for (; i < inF; i++) {
                    int nibble = (qweightInts[pr * inF + i] >> bitShift) & quantMask;
                    dot += lookupFp32[i * numLevels + nibble] * inputVec[i];
                }
                outputVec[j] = dot;
            }
        }

        // Sparse: scatter-multiply
        if (weightNnz != null && denseIdx != null && densePtr != null) {
            for (int i = 0; i < outF; i++) {
                float acc = 0f;
                for (int k = densePtr[i]; k < densePtr[i + 1]; k++) {
                    int col = denseIdx[k];
                    if (col < inF)
                        acc += weightNnz[k] * inputVec[col];
                }
                outputVec[i] += acc;
            }
        }
    }

    // ── Sparsity Statistics ───────────────────────────────────────────────────

    /**
     * Computes sparsity statistics from CSR structure.
     */
    public record SparsityStats(int totalWeights, int nnz, double sparsityPct) {
        public String format() {
            return "nnz=%,d / %,d total (%.2f%% sparse)".formatted(nnz, totalWeights, sparsityPct);
        }
    }

    public SparsityStats computeSparsity(int[] densePtr, int outF, int inF) {
        int nnz = densePtr.length > outF ? densePtr[outF] : 0;
        int total = outF * inF;
        return new SparsityStats(total, nnz, nnz * 100.0 / total);
    }

    // ── FP16 utility ──────────────────────────────────────────────────────────

    private static float fp16ToFloat32(short s) {
        int h = s & 0xFFFF;
        int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
        if (exp == 0)
            return Float.intBitsToFloat((sign << 31) | (mant == 0 ? 0 : ((mant << 13) | ((127 - 15 - 1) << 23))));
        if (exp == 31)
            return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (mant << 13));
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    public static void printCapabilities() {
        System.out.println("=== SqueezeLLM Dequantizer ===");
        System.out.printf("Format: dense (non-uniform LUT) + sparse CSR outliers%n");
        System.out.printf("Dense:  per-column codebook [inF × 2^bits]%n");
        System.out.printf("Sparse: CSR (dense_ptr, dense_idx, weight_nnz)%n");
        System.out.printf("SIMD:   %s (%d float lanes)%n", FLOAT_SPECIES.toString(), F_LANES);
    }
}
