package tech.kayys.gollek.quantizer.turboquant;

import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GGUF/GGML Dequantization Engine — JDK 25 Vector API.
 *
 * GGUF (GPT-Generated Unified Format) uses block-based quantization where
 * weights are organized into fixed-size blocks, each with its own scale.
 * This is distinct from tensor-level group quantization (GPTQ/AWQ).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ GGUF Quantization Types Supported │
 * ├──────────────┬──────────┬───────────────────────────────────────────────┤
 * │ Type │ Block sz │ Description │
 * ├──────────────┼──────────┼───────────────────────────────────────────────┤
 * │ Q4_0 │ 18 B │ 4-bit, 32 weights/block, 1×FP16 scale │
 * │ Q4_1 │ 20 B │ Q4_0 + FP16 min offset │
 * │ Q5_0 │ 22 B │ 5-bit, 32 weights/block, 1×FP16 scale │
 * │ Q5_1 │ 24 B │ Q5_0 + FP16 min offset │
 * │ Q8_0 │ 34 B │ 8-bit, 32 weights/block, 1×FP16 scale │
 * │ Q2_K │ 84 B │ 2-bit super-blocks, 256 weights │
 * │ Q3_K_S/M/L │ 110 B │ 3-bit super-blocks, 256 weights │
 * │ Q4_K_S/M │ 144 B │ 4-bit super-blocks, 256 weights (most used) │
 * │ Q5_K_S/M │ 176 B │ 5-bit super-blocks, 256 weights │
 * │ Q6_K │ 210 B │ 6-bit super-blocks, 256 weights │
 * └──────────────┴──────────┴───────────────────────────────────────────────┘
 *
 * Block layout for Q4_K_M (most popular for LLMs):
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Q4_K Super-Block (144 bytes, 256 elements): │
 * │ [2×FP16 d,dmin] [12×u8 scales+mins] [128×u8 quantized nibbles] │
 * │ ├── d : super-block scale (FP16, 2 bytes) │
 * │ ├── dmin : super-block min scale (FP16, 2 bytes) │
 * │ ├── scales: 12 bytes encoding 8 sub-block scales and 8 mins │
 * │ └── qs : 128 bytes of packed INT4 (256 values × 4 bits) │
 * │ │
 * │ Dequant: w[i] = d × scale[sub] × q[i] - dmin × min[sub] │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Vector API strategy for Q4_K_M:
 * - Load 128 bytes (256 nibbles) per super-block
 * - IntVector: unpack 8 nibbles per INT32 in parallel
 * - FloatVector: apply sub-block scale and min in SIMD batches
 * - Process F_LANES elements per SIMD iteration
 */
public class GGUFDequantizer {

    private static final Logger log = LoggerFactory.getLogger(GGUFDequantizer.class);

    // Use the preferred vector species for the current platform
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final int F_LANES = FLOAT_SPECIES.length();

    /** GGUF quantization type identifiers (from ggml.h) */
    public enum GGMLType {
        Q4_0(2, 18, 32, 4, false),
        Q4_1(3, 20, 32, 4, true),
        Q5_0(6, 22, 32, 5, false),
        Q5_1(7, 24, 32, 5, true),
        Q8_0(8, 34, 32, 8, false),
        Q2_K(10, 84, 256, 2, true),
        Q3_K(11, 110, 256, 3, false),
        Q4_K(12, 144, 256, 4, true),
        Q5_K(13, 176, 256, 5, true),
        Q6_K(14, 210, 256, 6, false),
        F16(1, 2, 1, 16, false),
        F32(0, 4, 1, 32, false);

        public final int ggmlId;
        public final int blockBytes;
        public final int blockSize; // weights per block / super-block
        public final int bits;
        public final boolean hasMin; // whether a separate min-scale is stored

        GGMLType(int id, int blockBytes, int blockSize, int bits, boolean hasMin) {
            this.ggmlId = id;
            this.blockBytes = blockBytes;
            this.blockSize = blockSize;
            this.bits = bits;
            this.hasMin = hasMin;
        }

        public double compressionRatio() {
            return 32.0 / bits;
        }

        public static GGMLType fromId(int id) {
            for (GGMLType t : values())
                if (t.ggmlId == id)
                    return t;
            return F32;
        }
    }

    // ── Q4_0: Simple 4-bit, 32 weights/block ─────────────────────────────────

    /**
     * Dequantizes Q4_0 block-quantized data.
     *
     * Block layout (18 bytes):
     * [2 bytes FP16 scale d][16 bytes: 32 INT4 weights packed as nibbles]
     *
     * Formula: w[i] = d × (q[i] − 8) (subtract 8 to center at 0)
     *
     * @param rawBlocks   raw block bytes from GGUF file
     * @param numElements number of FP32 elements to output
     * @param output      pre-allocated FP32 output
     */
    public void dequantQ4_0(byte[] rawBlocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32;
        final int BLOCK_BYTES = 18;
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < numBlocks; b++) {
            int blockOff = b * BLOCK_BYTES;
            int outBase = b * BLOCK_SIZE;

            // Read FP16 scale
            short dBits = (short) (((rawBlocks[blockOff + 1] & 0xFF) << 8)
                    | (rawBlocks[blockOff] & 0xFF));
            float d = fp16ToFloat(dBits);

            // Unpack 32 nibbles from 16 bytes
            int nib = 0;
            while (nib < BLOCK_SIZE) {
                int j = nib / 2;
                int packed = rawBlocks[blockOff + 2 + j] & 0xFF;
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;

                int outIdx = outBase + nib;
                if (outIdx < numElements)
                    output[outIdx] = d * lo;
                if (outIdx + 1 < numElements)
                    output[outIdx + 1] = d * hi;
                nib += 2;
            }
        }
    }

    // ── Q8_0: 8-bit, 32 weights/block ────────────────────────────────────────

    /**
     * Dequantizes Q8_0 block-quantized data with Vector API inner loop.
     *
     * Block layout (34 bytes):
     * [2 bytes FP16 scale d][32 bytes: INT8 weights]
     *
     * Formula: w[i] = d × q[i]
     */
    public void dequantQ8_0(byte[] rawBlocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32;
        final int BLOCK_BYTES = 34;
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < numBlocks; b++) {
            int blockOff = b * BLOCK_BYTES;
            int outBase = b * BLOCK_SIZE;

            short dBits = (short) (((rawBlocks[blockOff + 1] & 0xFF) << 8)
                    | (rawBlocks[blockOff] & 0xFF));
            float d = fp16ToFloat(dBits);

            // SIMD-friendly: process F_LANES INT8 values at once
            int i = 0;
            for (; i <= BLOCK_SIZE - F_LANES && outBase + i + F_LANES <= numElements; i += F_LANES) {
                float[] vals = new float[F_LANES];
                for (int lane = 0; lane < F_LANES; lane++) {
                    vals[lane] = (float) rawBlocks[blockOff + 2 + i + lane]; // signed INT8
                }
                FloatVector vq = FloatVector.fromArray(FLOAT_SPECIES, vals, 0);
                FloatVector vd = FloatVector.broadcast(FLOAT_SPECIES, d);
                vq.mul(vd).intoArray(output, outBase + i);
            }
            // Scalar tail
            for (; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                output[outBase + i] = d * rawBlocks[blockOff + 2 + i];
            }
        }
    }

    // ── Q4_K: 4-bit Super-Blocks (most popular for 7B/13B LLMs) ─────────────

    /**
     * Dequantizes Q4_K_M super-block format — the most commonly used GGUF type.
     *
     * Super-block layout (144 bytes, 256 elements, 8 sub-blocks of 32):
     * [2B FP16 d][2B FP16 dmin][12B scales+mins][128B packed INT4]
     *
     * The 12-byte scales/mins block encodes:
     * - 8 × 6-bit sub-block scales (bits[5:0] of each scale byte)
     * - 8 × 6-bit sub-block mins (packed into upper bits)
     *
     * Formula: w[i] = d × sc[sub] × q[i] − dmin × mn[sub]
     * where sc[sub], mn[sub] are the sub-block scale and min.
     *
     * @param rawBlocks   raw bytes from GGUF tensor data
     * @param numElements total FP32 elements to write
     * @param output      output array
     */
    public void dequantQ4K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_SIZE = 256;
        final int SUPER_BLOCK_BYTES = 144;
        final int NUM_SUB = 8; // 8 sub-blocks per super-block
        final int SUB_SIZE = 32;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            // Read super-block scale (FP16) and min-scale (FP16)
            float d = fp16ToFloat(readFp16(rawBlocks, sbOff));
            float dmin = fp16ToFloat(readFp16(rawBlocks, sbOff + 2));

            // Decode 8 sub-block scales and 8 mins from 12 bytes
            float[] sc = new float[NUM_SUB];
            float[] mn = new float[NUM_SUB];
            decodeQ4KScales(rawBlocks, sbOff + 4, d, dmin, sc, mn);

            // Dequantize 256 nibbles from 128 bytes
            int qsOff = sbOff + 16; // nibble data starts at byte 16

            for (int sub = 0; sub < NUM_SUB; sub++) {
                int subOutBase = outBase + sub * SUB_SIZE;
                float scale = sc[sub];
                float minV = mn[sub];

                // Two halves of 16 nibbles each:
                // First 16 nibbles: low bits of qs[0..15]
                // Second 16 nibbles: high bits of qs[0..15]
                int halfOff = qsOff + sub * 16; // low nibbles offset
                int hiOff = qsOff + 64 + sub * 16; // high nibbles offset

                for (int i = 0; i < 16 && subOutBase + i < numElements; i++) {
                    int bLo = rawBlocks[halfOff + i] & 0xFF;
                    int bHi = rawBlocks[hiOff + i] & 0xFF;
                    int lo = bLo & 0xF;
                    int hi = bHi & 0xF;

                    output[subOutBase + i] = scale * lo - minV;
                    if (subOutBase + i + 16 < numElements)
                        output[subOutBase + i + 16] = scale * hi - minV;
                }
            }
        }
    }

    /**
     * Decode Q4_K 12-byte scales+mins block into per-sub-block FP32 arrays.
     *
     * The 12 bytes encode:
     * Bytes 0-5: lower 4 bits of scale[0..7] and lower 4 bits of min[0..7]
     * Bytes 6-11: upper 2 bits completing each 6-bit scale and min value
     */
    private void decodeQ4KScales(byte[] data, int off,
            float d, float dmin,
            float[] sc, float[] mn) {
        for (int i = 0; i < 8; i++) {
            // Lower 4 bits come from bytes 0-7 (alternating scale/min)
            int lo = data[off + i] & 0xFF;
            int scLo = lo & 0x3F; // 6-bit scale (lower 6 bits from byte i)
            int mnLo = (data[off + i + 4] >> 0) & 0x3F;
            // Extend with bits from bytes 8-11
            sc[i] = d * scLo;
            mn[i] = dmin * mnLo;
        }
    }

    // ── Q5_K: 5-bit Super-Blocks ──────────────────────────────────────────────

    /**
     * Dequantizes Q5_K format.
     *
     * Like Q4_K but with an extra high-bit byte per sub-block providing
     * the 5th bit for each quantized value.
     *
     * Super-block layout (176 bytes, 256 elements):
     * [2B d][2B dmin][12B scales][32B high bits][128B low nibbles]
     *
     * w[i] = d × sc[sub] × q5[i] - dmin × mn[sub]
     * where q5[i] = (hi_bit[i] << 4) | lo_nibble[i] (0..31)
     */
    public void dequantQ5K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_BYTES = 176;
        final int SUPER_BLOCK_SIZE = 256;
        final int NUM_SUB = 8;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            float d = fp16ToFloat(readFp16(rawBlocks, sbOff));
            float dmin = fp16ToFloat(readFp16(rawBlocks, sbOff + 2));

            float[] sc = new float[NUM_SUB];
            float[] mn = new float[NUM_SUB];
            decodeQ4KScales(rawBlocks, sbOff + 4, d, dmin, sc, mn);

            int hiBitsOff = sbOff + 16; // 32 bytes of high bits
            int qsOff = sbOff + 48; // 128 bytes of low nibbles

            for (int sub = 0; sub < NUM_SUB; sub++) {
                int subOut = outBase + sub * 32;
                float scale = sc[sub], minV = mn[sub];

                for (int i = 0; i < 32 && subOut + i < numElements; i++) {
                    int byteIdx = sub * 4 + i / 8;
                    int bit = (rawBlocks[hiBitsOff + byteIdx] >> (i % 8)) & 1;
                    int nibIdx = sub * 16 + i / 2;
                    int nibble = (rawBlocks[qsOff + nibIdx] >> ((i % 2) * 4)) & 0xF;
                    int q5 = (bit << 4) | nibble;
                    output[subOut + i] = scale * q5 - minV;
                }
            }
        }
    }

    // ── Q6_K: 6-bit Super-Blocks ──────────────────────────────────────────────

    /**
     * Dequantizes Q6_K (6-bit, highest quality GGUF format).
     *
     * Super-block layout (210 bytes, 256 elements):
     * [128B low nibbles][64B high 2-bits][16B scales (INT8)][2B FP16 d]
     *
     * w[i] = d × scale[sub] × q6[i]
     * where q6[i] = lo4_nibble[i] | (hi2_bit[i] << 4) (0..63, biased by -32)
     */
    public void dequantQ6K(byte[] rawBlocks, int numElements, float[] output) {
        final int SUPER_BLOCK_BYTES = 210;
        final int SUPER_BLOCK_SIZE = 256;
        final int NUM_SUB = 16;

        int numSuperBlocks = (numElements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;

        for (int sb = 0; sb < numSuperBlocks; sb++) {
            int sbOff = sb * SUPER_BLOCK_BYTES;
            int outBase = sb * SUPER_BLOCK_SIZE;

            // Q6_K layout: nibbles first, then high bits, then scales, then d
            int ql = sbOff; // 128 bytes: low 4-bit nibbles
            int qh = sbOff + 128; // 64 bytes: high 2-bit pairs
            int sc = sbOff + 192; // 16 bytes: INT8 sub-block scales
            int dOff = sbOff + 208; // 2 bytes: FP16 super-block scale

            float d = fp16ToFloat(readFp16(rawBlocks, dOff));

            for (int i = 0; i < SUPER_BLOCK_SIZE && outBase + i < numElements; i++) {
                int sub = i / 16;
                float subScale = d * rawBlocks[sc + sub]; // INT8 scale

                // Extract 6-bit value: lo4 from ql, hi2 from qh
                int qIdx = i / 2;
                int lo4 = (rawBlocks[ql + qIdx] >> ((i % 2) * 4)) & 0xF;
                int hiIdx = i / 4;
                int hi2 = (rawBlocks[qh + hiIdx] >> ((i % 4) * 2)) & 0x3;
                int q6 = lo4 | (hi2 << 4);
                output[outBase + i] = subScale * (q6 - 32); // bias by 32
            }
        }
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────

    /**
     * Dispatch to the correct dequant implementation for a given GGML type.
     */
    public void dequantize(GGMLType type, byte[] rawBlocks, int numElements, float[] output) {
        switch (type) {
            case Q4_0 -> dequantQ4_0(rawBlocks, numElements, output);
            case Q4_1 -> dequantQ4_1(rawBlocks, numElements, output);
            case Q5_0 -> dequantQ5_0(rawBlocks, numElements, output);
            case Q8_0 -> dequantQ8_0(rawBlocks, numElements, output);
            case Q4_K -> dequantQ4K(rawBlocks, numElements, output);
            case Q5_K -> dequantQ5K(rawBlocks, numElements, output);
            case Q6_K -> dequantQ6K(rawBlocks, numElements, output);
            case F16 -> dequantF16(rawBlocks, numElements, output);
            case F32 -> dequantF32(rawBlocks, numElements, output);
            default -> throw new UnsupportedOperationException(
                    "GGML type not yet implemented: " + type);
        }
        log.debug("Dequantized {} elements using {}", numElements, type);
    }

    // ── Q4_1, Q5_0 (with min offset) ─────────────────────────────────────────

    private void dequantQ4_1(byte[] blocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32, BLOCK_BYTES = 20;
        int n = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int b = 0; b < n; b++) {
            int bOff = b * BLOCK_BYTES, outBase = b * BLOCK_SIZE;
            float d = fp16ToFloat(readFp16(blocks, bOff));
            float m = fp16ToFloat(readFp16(blocks, bOff + 2));
            for (int i = 0; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                int packed = blocks[bOff + 4 + i / 2] & 0xFF;
                int q = (i % 2 == 0) ? (packed & 0xF) : (packed >> 4);
                output[outBase + i] = d * q + m;
            }
        }
    }

    private void dequantQ5_0(byte[] blocks, int numElements, float[] output) {
        final int BLOCK_SIZE = 32, BLOCK_BYTES = 22;
        int n = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int b = 0; b < n; b++) {
            int bOff = b * BLOCK_BYTES, outBase = b * BLOCK_SIZE;
            float d = fp16ToFloat(readFp16(blocks, bOff));
            // 4 bytes high bits
            int qhInt = ((blocks[bOff + 2] & 0xFF)) | ((blocks[bOff + 3] & 0xFF) << 8)
                    | ((blocks[bOff + 4] & 0xFF) << 16) | ((blocks[bOff + 5] & 0xFF) << 24);
            for (int i = 0; i < BLOCK_SIZE && outBase + i < numElements; i++) {
                int nibIdx = bOff + 6 + i / 2;
                int lo4 = (i % 2 == 0) ? (blocks[nibIdx] & 0xF) : ((blocks[nibIdx] >> 4) & 0xF);
                int hi1 = (qhInt >> i) & 1;
                int q5 = lo4 | (hi1 << 4);
                output[outBase + i] = d * (q5 - 16);
            }
        }
    }

    private void dequantF16(byte[] blocks, int numElements, float[] output) {
        for (int i = 0; i < numElements; i++) {
            output[i] = fp16ToFloat(readFp16(blocks, i * 2));
        }
    }

    private void dequantF32(byte[] blocks, int numElements, float[] output) {
        for (int i = 0; i < numElements; i++) {
            int bits = ((blocks[i * 4] & 0xFF)) | ((blocks[i * 4 + 1] & 0xFF) << 8)
                    | ((blocks[i * 4 + 2] & 0xFF) << 16) | ((blocks[i * 4 + 3] & 0xFF) << 24);
            output[i] = Float.intBitsToFloat(bits);
        }
    }

    // ── FP16 Utility ──────────────────────────────────────────────────────────

    private static short readFp16(byte[] data, int off) {
        return (short) (((data[off + 1] & 0xFF) << 8) | (data[off] & 0xFF));
    }

    private static float fp16ToFloat(short fp16) {
        int h = fp16 & 0xFFFF;
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
            return Float.intBitsToFloat((sign << 31) | (mant == 0 ? 0 : ((mant << 13) | ((127 - 15 - 1) << 23))));
        if (exp == 31)
            return Float.intBitsToFloat((sign << 31) | 0x7F800000 | (mant << 13));
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    public static void printCapabilities() {
        System.out.println("=== GGUF Dequantizer ===");
        System.out.printf("Supported types: Q4_0 Q4_1 Q5_0 Q8_0 Q4_K Q5_K Q6_K F16 F32%n");
        System.out.printf("SIMD: %s (%d float lanes)%n", FLOAT_SPECIES.toString(), F_LANES);
    }
}
