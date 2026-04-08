package tech.kayys.gollek.quantizer.turboquant;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HQQ (Half-Quadratic Quantization) Dequantization Engine — JDK 25 Vector API.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ HQQ Algorithm Overview │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │ HQQ minimizes a Half-Quadratic (HQ) objective, which is robust to │
 * │ outliers. Unlike GPTQ (full Hessian) or AWQ (activation channels), │
 * │ HQQ works DIRECTLY on the weight distribution, no calibration data. │
 * │ │
 * │ HQ objective: min_{W_q} ρ(W − dequant(W_q, scale, zero)) │
 * │ where ρ is a half-quadratic penalty (robust to outliers)│
 * │ │
 * │ Key properties: │
 * │ - Calibration-free (no data needed, instant quantization) │
 * │ - Per-axis (per-column or per-row) or per-group scales │
 * │ - Asymmetric: scale and zero-point both learned │
 * │ - Compatible with 1, 2, 3, 4, 8 bit │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * HQQ Tensor Layout (hqq library, PyTorch):
 * ──────────────────────────────────────────────────────────────────────────
 * layer.W : INT32 packed weights [outF/pack, inF] (same as GPTQ)
 * layer.scale : FP16 scales [numGroups, outF]
 * layer.zero : FP16 zero-points [numGroups, outF]
 *
 * Note: HQQ uses SEPARATE FP16 tensors for scale AND zero (unlike GPTQ
 * which packs zeros into INT32). Both scale and zero are stored as FP16.
 *
 * Dequantization formula:
 * w_fp32[i,j] = scale[g,j] × q[i,j] + zero[g,j]
 * where g = i / groupSize
 *
 * Note the sign convention: HQQ adds zero-point (additive), while GPTQ
 * subtracts zero-point. This is equivalent to GPTQ with negated zeros.
 *
 * Axis options:
 * HQQ supports quantization along different axes:
 * axis=0: groups along INPUT dimension (default, like AWQ/AutoRound)
 * axis=1: groups along OUTPUT dimension (like GPTQ)
 *
 * The packing is identical to GPTQ for INT4:
 * W[pr, c] packs 8 output-feature weights for input col c.
 */
public class HQQDequantizer {

    private static final Logger log = LoggerFactory.getLogger(HQQDequantizer.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    /** HQQ quantization axis */
    public enum QuantAxis {
        /** Groups along INPUT dimension (axis=0, default) */
        INPUT,
        /** Groups along OUTPUT dimension (axis=1) */
        OUTPUT
    }

    private final int bits;
    private final int groupSize;
    private final QuantAxis axis;
    private final int packFactor;
    private final int quantMask;

    public HQQDequantizer(int bits, int groupSize, QuantAxis axis) {
        this.bits = bits;
        this.groupSize = groupSize;
        this.axis = axis;
        this.packFactor = 32 / bits;
        this.quantMask = (1 << bits) - 1;
        log.info("HQQDequantizer: bits={}, groupSize={}, axis={}, SIMD={}×FP32",
                bits, groupSize, axis, F_LANES);
    }

    // ── Main Dequantization ───────────────────────────────────────────────────

    /**
     * Dequantizes HQQ layer weights to FP32.
     *
     * @param qweightInts packed INT32 [outF/pack, inF] (same packing as GPTQ)
     * @param scalesFp32  FP32 scales [numGroups, outF] (pre-converted from FP16)
     * @param zerosFp32   FP32 zeros [numGroups, outF] (pre-converted from FP16)
     * @param inF         input features
     * @param outF        output features
     * @param output      FP32 output [outF, inF] (row = output feature)
     */
    public void dequantize(
            int[] qweightInts,
            float[] scalesFp32,
            float[] zerosFp32,
            int inF,
            int outF,
            float[] output) {
        if (axis == QuantAxis.OUTPUT) {
            dequantAxisOutput(qweightInts, scalesFp32, zerosFp32, inF, outF, output);
        } else {
            dequantAxisInput(qweightInts, scalesFp32, zerosFp32, inF, outF, output);
        }
    }

    // ── Axis=0 (groups over INPUT dimension) ──────────────────────────────────

    /**
     * HQQ dequantization with groups over the INPUT dimension.
     *
     * scale/zero shape: [numGroups_of_inF, outF]
     * w[i,j] = scale[g_i, j] × q[i,j] + zero[g_i, j]
     * g_i = i / groupSize
     *
     * Note the HQQ additive zero convention (opposite sign to GPTQ).
     */
    private void dequantAxisInput(
            int[] qweight, float[] scales, float[] zeros,
            int inF, int outF, float[] output) {
        int numGroups = (inF + groupSize - 1) / groupSize;
        int packedRows = outF / packFactor;

        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;
                int bitShift = b * bits;
                int outRow = j * inF;

                int i = 0;
                for (; i <= inF - F_LANES; i += F_LANES) {
                    int g = i / groupSize;
                    // HQQ scale/zero layout: [numGroups, outF]
                    // Indexed as: scales[g * outF + j]
                    float s = scales[g * outF + j];
                    float z = zeros[g * outF + j];

                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        qVals[lane] = (float) ((qweight[pr * inF + i + lane] >> bitShift) & quantMask);
                    }

                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vs = FloatVector.broadcast(FLOAT_SPECIES, s);
                    FloatVector vz = FloatVector.broadcast(FLOAT_SPECIES, z);

                    // HQQ: w = scale × q + zero (additive zero!)
                    vq.fma(vs, vz).intoArray(output, outRow + i);
                }
                for (; i < inF; i++) {
                    int g = i / groupSize;
                    int qVal = (qweight[pr * inF + i] >> bitShift) & quantMask;
                    output[outRow + i] = scales[g * outF + j] * qVal + zeros[g * outF + j];
                }
            }
        }
    }

    // ── Axis=1 (groups over OUTPUT dimension, like GPTQ) ──────────────────────

    /**
     * HQQ dequantization with groups over the OUTPUT dimension.
     *
     * scale/zero shape: [numGroups_of_outF, inF]
     * w[i,j] = scale[g_j, i] × q[i,j] + zero[g_j, i]
     * g_j = j / groupSize
     */
    private void dequantAxisOutput(
            int[] qweight, float[] scales, float[] zeros,
            int inF, int outF, float[] output) {
        int numGroupsOut = (outF + groupSize - 1) / groupSize;
        int packedRows = outF / packFactor;

        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;
                int bitShift = b * bits;
                int gj = j / groupSize;
                int outRow = j * inF;

                int i = 0;
                for (; i <= inF - F_LANES; i += F_LANES) {
                    float[] qVals = new float[F_LANES];
                    float[] sVals = new float[F_LANES];
                    float[] zVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++) {
                        int col = i + lane;
                        qVals[lane] = (float) ((qweight[pr * inF + col] >> bitShift) & quantMask);
                        sVals[lane] = scales[gj * inF + col];
                        zVals[lane] = zeros[gj * inF + col];
                    }
                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vs = FloatVector.fromArray(FLOAT_SPECIES, sVals, 0);
                    FloatVector vz = FloatVector.fromArray(FLOAT_SPECIES, zVals, 0);
                    vq.fma(vs, vz).intoArray(output, outRow + i);
                }
                for (; i < inF; i++) {
                    int qVal = (qweight[pr * inF + i] >> bitShift) & quantMask;
                    output[outRow + i] = scales[gj * inF + i] * qVal + zeros[gj * inF + i];
                }
            }
        }
    }

    // ── Fused MatVec for inference ────────────────────────────────────────────

    /**
     * Fused dequant + matmul: outputVec[j] = Σ_i w[i,j] × inputVec[i]
     * HQQ version: w[i,j] = scale[g,j] × q + zero[g,j]
     */
    public void dequantMatVec(
            int[] qweight, float[] scales, float[] zeros,
            float[] inputVec, float[] outputVec,
            int inF, int outF) {
        int numGroups = (inF + groupSize - 1) / groupSize;
        int packedRows = outF / packFactor;
        java.util.Arrays.fill(outputVec, 0f);

        for (int pr = 0; pr < packedRows; pr++) {
            for (int b = 0; b < packFactor; b++) {
                int j = pr * packFactor + b;
                if (j >= outF)
                    break;
                int bitShift = b * bits;
                float dot = 0f;
                int i = 0;
                FloatVector acc = FloatVector.zero(FLOAT_SPECIES);

                for (; i <= inF - F_LANES; i += F_LANES) {
                    int g = i / groupSize;
                    float s = scales[g * outF + j], z = zeros[g * outF + j];
                    float[] qVals = new float[F_LANES];
                    for (int lane = 0; lane < F_LANES; lane++)
                        qVals[lane] = (float) ((qweight[pr * inF + i + lane] >> bitShift) & quantMask);
                    FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, qVals, 0);
                    FloatVector vx = FloatVector.fromArray(FLOAT_SPECIES, inputVec, i);
                    FloatVector vs = FloatVector.broadcast(FLOAT_SPECIES, s);
                    FloatVector vz = FloatVector.broadcast(FLOAT_SPECIES, z);
                    // w = s*q + z; out += w * x = (s*q + z)*x = s*q*x + z*x
                    acc = vq.fma(vs, vz).fma(vx, acc);
                }
                dot = acc.reduceLanes(VectorOperators.ADD);
                for (; i < inF; i++) {
                    int g = i / groupSize;
                    int q = (qweight[pr * inF + i] >> bitShift) & quantMask;
                    dot += (scales[g * outF + j] * q + zeros[g * outF + j]) * inputVec[i];
                }
                outputVec[j] = dot;
            }
        }
    }

    public static void printCapabilities() {
        System.out.println("=== HQQ Dequantizer ===");
        System.out.printf("Axes: INPUT (axis=0), OUTPUT (axis=1)%n");
        System.out.printf("Formula: w = scale × q + zero  (additive zero-point)%n");
        System.out.printf("SIMD: %s (%d float lanes)%n", FLOAT_SPECIES.toString(), F_LANES);
    }
}
