package tech.kayys.gollek.quantizer.autoround;

import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AutoRound Dequantization Engine — JDK 25 Vector API (SIMD).
 *
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │ Comparison of Dequantization Formulas │
 * ├─────────────────┬────────────────────────────────────────────────────────┤
 * │ GPTQ │ w = (q − zero_packed) × scale_fp16 │
 * │ │ groups over OUTPUT dim, scale/zero packed INT32 │
 * ├─────────────────┼────────────────────────────────────────────────────────┤
 * │ AWQ │ w = (q − zero_packed) × scale_fp16 │
 * │ │ groups over INPUT dim, qweight transposed │
 * ├─────────────────┼────────────────────────────────────────────────────────┤
 * │ AutoRound │ w = (q − zp_plain) × scale_fp32 │
 * │ (native) │ groups over INPUT dim, zp is plain INT32 (not packed) │
 * │ │ scale is FP32 (not FP16), qweight same as GPTQ layout │
 * ├─────────────────┼────────────────────────────────────────────────────────┤
 * │ AutoRound │ w = (q − zero_packed + 1) × scale_fp16 │
 * │ (GPTQ-compat) │ identical to GPTQ formula after zp unpack │
 * └─────────────────┴────────────────────────────────────────────────────────┘
 *
 * AutoRound qweight Packing (AUTOROUND_NATIVE and GPTQ_COMPAT):
 * ──────────────────────────────────────────────────────────────────────────
 * Shape: [outF / packFactor, inF]
 * Each INT32 word packs packFactor (=8 for 4-bit) consecutive OUTPUT
 * feature weights for the same INPUT feature column:
 *
 * qweight[pr, c] = q[pr*8+0, c] | (q[pr*8+1, c] << 4)
 * | (q[pr*8+2, c] << 8) | ... | (q[pr*8+7, c] << 28)
 *
 * where pr = output row index / packFactor, c = input column index.
 *
 * AutoRound Scale Layout:
 * ──────────────────────────────────────────────────────────────────────────
 * Shape: [outF, inF / groupSize] — indexed as scale[j, g]
 * One scale per (output feature j, input group g).
 * g = input_index / groupSize
 *
 * AutoRound Zero-Point Layout (native):
 * ──────────────────────────────────────────────────────────────────────────
 * Shape: [outF, inF / groupSize] — same as scale, indexed as zp[j, g]
 * Stored as plain INT32 values (each zp occupies 4 bytes).
 * NOT packed — one INT32 per group per output feature.
 *
 * ITREX Packing:
 * ──────────────────────────────────────────────────────────────────────────
 * Intel Extension for PyTorch (ITREX) uses row-major packing:
 * qweight[r, pc] where pc = input column / packFactor.
 * Bit order within INT32 is also reversed (MSB = first element).
 * We de-interleave to standard layout before dequantizing.
 *
 * Vector API Strategy:
 * ──────────────────────────────────────────────────────────────────────────
 * Outer loop : packed output rows (pr = j/packFactor)
 * Inner loop : input features (i) in SIMD chunks of F_LANES
 * Per SIMD : FloatVector.sub(vz).mul(vs) = (q − zp) × scale
 * FMA path : For inference, fuses dequant + matmul into one pass
 */
public class AutoRoundDequantizer {

    private static final Logger log = LoggerFactory.getLogger(AutoRoundDequantizer.class);

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();
    private static final int I_LANES = INT_SPECIES.length();

    private final AutoRoundConfig config;
    private final int packFactor;
    private final int quantMask;

    public AutoRoundDequantizer(AutoRoundConfig config) {
        this.config = config;
        this.packFactor = config.packFactor();
        this.quantMask = config.quantMask();

        log.info("AutoRoundDequantizer: bits={}, pack={}, mask=0x{}, packFmt={}, " +
                "scaleDtype={}, SIMD={}×FP32 ({})",
                config.bits(), packFactor, Integer.toHexString(quantMask),
                config.packFormat(), config.scaleDtype(),
                F_LANES, FLOAT_SPECIES.toString());
    }

    // ── Main Entry Point ──────────────────────────────────────────────────────

    /**
     * Dequantizes a full AutoRound layer to FP32.
     *
     * Dispatches to the correct implementation based on
     * {@link AutoRoundConfig#packFormat()}.
     *
     * @param qweightInts packed INT32 weights [outF/pack, inF]
     * @param scalesFp32  FP32 scales [outF, inF/group] (pre-converted)
     * @param zpInts      INT32 zero-points [outF, inF/group] or null
     * @param inF         input feature count
     * @param outF        output feature count
     * @param output      pre-allocated FP32 output [outF × inF], row-major
     */
    public void dequantize(
            int[] qweightInts,
            float[] scalesFp32,
            int[] zpInts, // nullable — symmetric uses implicit zp
            int inF,
            int outF,
            float[] output) {
        // Build effective zero-point array (FP32) — once, before dispatch
        float[] zeros = buildZeroArray(zpInts, outF, inF);

        switch (config.packFormat()) {
            case AUTOROUND_NATIVE, GPTQ_COMPAT ->
                dequantizeNative(qweightInts, scalesFp32, zeros, inF, outF, output);
            case ITREX ->
                dequantizeItrex(qweightInts, scalesFp32, zeros, inF, outF, output);
        }
    }

    // ── Native / GPTQ-Compat Dequantization ──────────────────────────────────

    /**
     * AutoRound native and GPTQ-compat dequantization.
     *
     * Weight packing: qweight[pr, c] — pr = j/pack, c = input col
     * Unpack bit b of packed row pr → output row j = pr*pack + b
     *
     * For each output feature j and each input feature i:
     * g = i / groupSize
     * q_val = (qweight[j/pack, i] >> ((j % pack)*bits)) & mask
     * w[j, i] = (q_val − zp[j, g]) × scale[j, g]
     *
     * SIMD inner loop processes F_LANES input features per iteration.
     */
    private void dequantizeNative(
            int[] qweight, float[] scales, float[] zeros,
            int inF, int outF, float[] output) {
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        int packedRows = outF / packFactor; // number of INT32 rows

        for (int pr = 0; pr < packedRows; pr++) {
            // Process each bit-slice (sub-row) packed into this INT32 row
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b; // output feature index
                if (j >= outF)
                    break;

                int bitShift = b * config.bits();
                int outRow = j * inF; // row offset in output[]

                // Process input features in SIMD chunks
                int i = 0;
                for (; i <= inF - F_LANES; i += F_LANES) {
                    int g = i / config.groupSize(); // group for this input block
                    // NOTE: if F_LANES spans a group boundary we use g for the
                    // start of the block; a refined impl would split at the boundary.
                    // For typical groupSize=128 and F_LANES≤16, this is a non-issue.

                    // Unpack F_LANES quantized values
                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        // qweight[pr, i+lane] — row-major: pr*inF + (i+lane)
                        qVals[lane] = (float) ((qweight[pr * inF + i + lane] >> bitShift) & quantMask);
                    }

                    // scale[j, g] and zero[j, g] — both shaped [outF, numGroups]
                    float s = scales[j * numGroups + g];
                    float z = zeros[j * numGroups + g];

                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vs = FloatVector.broadcast(FLOAT_SPECIES, s);
                    FloatVector vz = FloatVector.broadcast(FLOAT_SPECIES, z);

                    // (q − z) × s
                    FloatVector result = vq.sub(vz).mul(vs);
                    result.intoArray(output, outRow + i);
                }

                // Scalar tail
                for (; i < inF; i++) {
                    int g = i / config.groupSize();
                    int qVal = (qweight[pr * inF + i] >> bitShift) & quantMask;
                    output[outRow + i] = (qVal - zeros[j * numGroups + g])
                            * scales[j * numGroups + g];
                }
            }
        }
    }

    // ── ITREX Packing (Intel Extension for PyTorch) ───────────────────────────

    /**
     * Intel ITREX dequantization.
     *
     * ITREX packs weight row-major: qweight[r, pc] where r = output row, pc =
     * col/pack.
     * Within each INT32, bit ordering is MSB-first (reversed from standard GPTQ).
     *
     * qweight[j, pc] = (q[j, pc*8+7] << 0) | (q[j, pc*8+6] << 4) | ...
     * MSB = first element (bit 31..28), LSB = last (bits 3..0)
     *
     * We de-interleave to standard layout and then call dequantizeNative.
     */
    private void dequantizeItrex(
            int[] qweight, float[] scales, float[] zeros,
            int inF, int outF, float[] output) {
        log.debug("ITREX dequant: reordering {}×{} weight", outF, inF);

        int packedCols = inF / packFactor;
        int[] reordered = new int[outF / packFactor * inF]; // standard layout target

        for (int j = 0; j < outF; j++) {
            int pr = j / packFactor; // packed row in output layout
            int b = j % packFactor; // bit-slice in output layout

            for (int pc = 0; pc < packedCols; pc++) {
                // ITREX: qweight[j, pc] — row index = j, packed col = pc
                int itrexWord = qweight[j * packedCols + pc];

                // ITREX bit order: element k is at bit position (pack-1-k)*bits from MSB
                // → at bit position (packFactor-1-b)*bits from LSB in the itrex word
                int itrexBitShift = (packFactor - 1 - b) * config.bits();
                int qVal = (itrexWord >> itrexBitShift) & quantMask;

                // Store into standard layout: reordered[pr, pc*pack + b]
                // We need to spread 1 value per output feature over packed cols:
                // Standard: reordered[pr * inF + (pc*pack + b)] holds the b-th
                // bit-slice value for input block pc*pack
                // Actually for standard layout we pack by output dimension:
                // reordered[pr * inF + c] with bit-shift (j%pack)*bits
                // Since this is complex, we write directly to standard format:
                int stdCol = pc * packFactor + b; // effective input index
                if (stdCol < inF) {
                    // Pack into the correct standard-layout INT32 word
                    int stdPr = j / packFactor;
                    int stdBit = (j % packFactor) * config.bits();
                    reordered[stdPr * inF + stdCol] |= (qVal & quantMask) << stdBit;
                }
            }
        }

        dequantizeNative(reordered, scales, zeros, inF, outF, output);
    }

    // ── Fused Dequant + MatVec Inference ─────────────────────────────────────

    /**
     * Fused dequantize-and-multiply: output_vec += W_dequant × input_vec
     *
     * Avoids materialising the full FP32 weight matrix.
     * Suitable for inference (single-token or small-batch generation).
     *
     * @param qweightInts packed INT32 [outF/pack, inF]
     * @param scalesFp32  FP32 scales [outF, inF/group]
     * @param zpInts      INT32 zp [outF, inF/group] or null
     * @param inputVec    FP32 input [inF]
     * @param outputVec   FP32 output [outF] — accumulated in-place
     * @param inF,        outF dimensions
     */
    public void dequantMatVec(
            int[] qweightInts,
            float[] scalesFp32,
            int[] zpInts,
            float[] inputVec,
            float[] outputVec,
            int inF, int outF) {
        float[] zeros = buildZeroArray(zpInts, outF, inF);
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        int packedRows = outF / packFactor;

        java.util.Arrays.fill(outputVec, 0f);

        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;

                int bitShift = b * config.bits();
                float dotProd = 0f;
                int i = 0;

                // SIMD dot product: accumulate (q − z) × scale × input
                float prevS = scalesFp32[j * numGroups + 0];
                float prevZ = zeros[j * numGroups + 0];

                FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
                for (; i <= inF - F_LANES; i += F_LANES) {
                    int g = i / config.groupSize();
                    float s = scalesFp32[j * numGroups + g];
                    float z = zeros[j * numGroups + g];

                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        qVals[lane] = (float) ((qweightInts[pr * inF + i + lane] >> bitShift) & quantMask);
                    }

                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, inputVec, i);
                    FloatVector vs = FloatVector.broadcast(FLOAT_SPECIES, s);
                    FloatVector vz = FloatVector.broadcast(FLOAT_SPECIES, z);

                    // acc += (q − z) × s × x
                    acc = vq.sub(vz).mul(vs).fma(vx, acc);
                }

                dotProd = acc.reduceLanes(VectorOperators.ADD);

                // Scalar tail
                for (; i < inF; i++) {
                    int g = i / config.groupSize();
                    int qVal = (qweightInts[pr * inF + i] >> bitShift) & quantMask;
                    float w = (qVal - zeros[j * numGroups + g]) * scalesFp32[j * numGroups + g];
                    dotProd += w * inputVec[i];
                }

                outputVec[j] = dotProd;
            }
        }
    }

    // ── Batched Inference ─────────────────────────────────────────────────────

    /**
     * Batched matrix multiply: output[b] = W_dequant × input[b]
     * for batchSize input vectors, processed together to amortise
     * the per-element dequant cost.
     *
     * @param qweightInts [outF/pack, inF]
     * @param scalesFp32  [outF, inF/group]
     * @param zpInts      [outF, inF/group] or null
     * @param inputs      [batchSize, inF] — row-major
     * @param outputs     [batchSize, outF] — row-major, overwritten
     * @param inF,        outF dimensions
     * @param batchSize   number of vectors to process
     */
    public void dequantMatMul(
            int[] qweightInts,
            float[] scalesFp32,
            int[] zpInts,
            float[] inputs,
            float[] outputs,
            int inF, int outF, int batchSize) {
        // Materialise the dequantized weight once and reuse across batch
        float[] wFp32 = new float[outF * inF];
        float[] zeros = buildZeroArray(zpInts, outF, inF);
        dequantizeNative(qweightInts, scalesFp32, zeros, inF, outF, wFp32);

        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();

        // SIMD batched matrix multiply
        for (int bIdx = 0; bIdx < batchSize; bIdx++) {
            int inOff = bIdx * inF;
            int outOff = bIdx * outF;

            for (int j = 0; j < outF; j++) {
                int wRowOff = j * inF;
                FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
                int k = 0;

                for (; k <= inF - F_LANES; k += F_LANES) {
                    FloatVector vw = FloatVector.fromArray(FLOAT_SPECIES, wFp32, wRowOff + k);
                    FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, inputs, inOff + k);
                    acc = vw.fma(vx, acc);
                }
                float sum = acc.reduceLanes(VectorOperators.ADD);
                for (; k < inF; k++)
                    sum += wFp32[wRowOff + k] * inputs[inOff + k];

                outputs[outOff + j] = sum;
            }
        }
    }

    // ── BF16 Scale Support ────────────────────────────────────────────────────

    /**
     * Converts BF16 (Brain Float 16) to FP32.
     *
     * BF16 is identical to the top 16 bits of FP32 (1 sign + 8 exp + 7 mantissa).
     * Conversion is a simple zero-fill of the lower 16 mantissa bits.
     *
     * @param bf16 raw BF16 bit pattern stored as short
     * @return equivalent FP32 value
     */
    public static float bf16ToFloat32(short bf16) {
        // BF16 bit pattern simply maps to upper 16 bits of a 32-bit float
        return Float.intBitsToFloat(((int) bf16 & 0xFFFF) << 16);
    }

    /** Bulk BF16 → FP32 conversion */
    public static float[] bf16ArrayToFloat32(short[] bf16) {
        float[] out = new float[bf16.length];
        for (int i = 0; i < bf16.length; i++)
            out[i] = bf16ToFloat32(bf16[i]);
        return out;
    }

    // ── Zero-Point Helpers ────────────────────────────────────────────────────

    /**
     * Builds the zero-point FP32 array.
     *
     * For AutoRound native: zpInts is plain INT32 [outF, inF/group].
     * Simply cast each element to float.
     *
     * For symmetric: fills with implicitZeroPoint (e.g., 8 for 4-bit).
     *
     * @param zpInts raw INT32 zero-points, or null for symmetric
     * @param outF   output features
     * @param inF    input features
     * @return FP32 zero-point array [outF, numGroups]
     */
    public float[] buildZeroArray(int[] zpInts, int outF, int inF) {
        int numGroups = (inF + config.groupSize() - 1) / config.groupSize();
        float[] zeros = new float[outF * numGroups];

        if (zpInts == null || !config.hasZeroPoint()) {
            // Symmetric: fill with implicit zero-point
            java.util.Arrays.fill(zeros, (float) config.implicitZeroPoint());
        } else {
            // AutoRound native: plain INT32 cast to float
            int copyLen = Math.min(zpInts.length, zeros.length);
            for (int i = 0; i < copyLen; i++) {
                zeros[i] = (float) zpInts[i];
            }
        }

        return zeros;
    }

    /**
     * Converts packed GPTQ-style INT32 zero-points to plain INT32 array.
     *
     * Used when loading GPTQ-compat AutoRound exports whose qzeros are
     * packed (same format as GPTQ). After unpacking we store as plain
     * INT32 in the AutoRoundLayer, consistent with native format.
     *
     * AutoRound GPTQ-compat zero convention: same as AutoGPTQ (+1 bias).
     *
     * @param packedZeros INT32 packed zeros [numGroups, outF/pack]
     * @param outF        output features
     * @param numGroups   number of groups
     * @return plain INT32 zeros [outF, numGroups]
     */
    public int[] unpackGptqCompatZeros(int[] packedZeros, int outF, int numGroups) {
        int packedCols = outF / packFactor;
        int[] plain = new int[outF * numGroups];

        for (int g = 0; g < numGroups; g++) {
            for (int pj = 0; pj < packedCols; pj++) {
                if (g * packedCols + pj >= packedZeros.length)
                    break;
                int packed = packedZeros[g * packedCols + pj];
                for (int b = 0; b < packFactor; b++) {
                    int j = pj * packFactor + b;
                    if (j < outF) {
                        // GPTQ-compat AutoRound: +1 bias (same as AutoGPTQ)
                        int raw = (packed >> (b * config.bits())) & quantMask;
                        // Store transposed: plain[j, g] = plain[j * numGroups + g]
                        plain[j * numGroups + g] = raw + 1;
                    }
                }
            }
        }

        return plain;
    }

    /**
     * Converts FP16 scales (GPTQ-compat) to FP32 and re-shapes from
     * [numGroups, outF] (GPTQ layout) to [outF, numGroups] (AutoRound layout).
     *
     * @param fp16Scales raw FP16 scale bits [numGroups × outF] — GPTQ layout
     * @param outF       output features
     * @param numGroups  number of groups
     * @return FP32 scales [outF × numGroups] — AutoRound layout
     */
    public float[] transposeFp16ScalesToFp32(short[] fp16Scales, int outF, int numGroups) {
        // GPTQ layout: scales[g, j] → index g * outF + j
        // AutoRound: scales[j, g] → index j * numGroups + g
        float[] out = new float[outF * numGroups];
        for (int g = 0; g < numGroups; g++) {
            for (int j = 0; j < outF; j++) {
                int gptqIdx = g * outF + j;
                int arIdx = j * numGroups + g;
                if (gptqIdx < fp16Scales.length) {
                    out[arIdx] = MemoryAllocator.fp16ToFloat32(fp16Scales[gptqIdx]);
                }
            }
        }
        return out;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    public static void printCapabilities() {
        System.out.println("=== AutoRound Vector API Capabilities ===");
        System.out.printf("Float SIMD : %s (%d lanes, %d-bit)%n",
                FLOAT_SPECIES.toString(), FLOAT_SPECIES.length(), FLOAT_SPECIES.vectorBitSize());
        System.out.printf("Int   SIMD : %s (%d lanes, %d-bit)%n",
                INT_SPECIES.toString(), INT_SPECIES.length(), INT_SPECIES.vectorBitSize());
        System.out.printf("ISA        : %s%n",
                FLOAT_SPECIES.vectorBitSize() >= 512 ? "AVX-512 (16 lanes)"
                        : FLOAT_SPECIES.vectorBitSize() >= 256 ? "AVX2/NEON+ (8 lanes)" : "SSE4/NEON (4 lanes)");
    }
}
