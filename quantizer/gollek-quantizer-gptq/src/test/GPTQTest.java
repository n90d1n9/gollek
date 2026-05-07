package tech.kayys.gollek.quantizer.gptq;

import org.junit.jupiter.api.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GPTQ components.
 * Tests run without a real model file — all data is synthetic.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class GPTQTest {

    // ── GPTQConfig Tests ───────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("GPTQConfig: 4-bit preset values")
    void testGptqConfig4Bit() {
        GPTQConfig cfg = GPTQConfig.gptq4bit();
        assertEquals(4, cfg.bits());
        assertEquals(128, cfg.groupSize());
        assertEquals(8, cfg.elementsPerInt32(), "4-bit: 32/4 = 8 elements per INT32");
        assertEquals(0xF, cfg.quantMask(), "4-bit mask = 0xF");
        assertEquals(8.0, cfg.compressionRatio(), 0.001);
        assertTrue(cfg.isValidBits());
    }

    @Test
    @Order(2)
    @DisplayName("GPTQConfig: 8-bit preset values")
    void testGptqConfig8Bit() {
        GPTQConfig cfg = GPTQConfig.gptq8bit();
        assertEquals(8, cfg.bits());
        assertEquals(4, cfg.elementsPerInt32(), "8-bit: 32/8 = 4 elements per INT32");
        assertEquals(0xFF, cfg.quantMask());
        assertEquals(4.0, cfg.compressionRatio(), 0.001);
    }

    @Test
    @Order(3)
    @DisplayName("GPTQConfig: 3-bit preset values")
    void testGptqConfig3Bit() {
        GPTQConfig cfg = GPTQConfig.gptq3bit();
        assertEquals(3, cfg.bits());
        assertEquals(0x7, cfg.quantMask());
        assertTrue(cfg.isValidBits());
    }

    @Test
    @Order(4)
    @DisplayName("GPTQConfig: zero-point offset asymmetric")
    void testZeroPointOffset() {
        GPTQConfig cfg4 = GPTQConfig.gptq4bit();
        assertEquals(8, cfg4.zeroPointOffset(), "4-bit asymmetric zeropoint = 2^3 = 8");

        GPTQConfig sym = new GPTQConfig(4, 128, false, true, false, "float32");
        assertEquals(0, sym.zeroPointOffset(), "symmetric: zeropoint = 0");
    }

    // ── FP16 Conversion Tests ──────────────────────────────────────────────────

    @Test
    @Order(5)
    @DisplayName("FP16 ↔ FP32: round-trip for common values")
    void testFp16RoundTrip() {
        float[] testValues = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 100.0f, 0.001f };

        for (float original : testValues) {
            short fp16 = MemoryAllocator.float32ToFp16(original);
            float reconstructed = MemoryAllocator.fp16ToFloat32(fp16);
            // FP16 has ~3 decimal digits of precision
            assertEquals(original, reconstructed, Math.abs(original) * 0.002 + 1e-6,
                    "Round-trip failed for: " + original);
        }
    }

    @Test
    @Order(6)
    @DisplayName("FP16: zero converts correctly")
    void testFp16Zero() {
        assertEquals(0.0f, MemoryAllocator.fp16ToFloat32((short) 0));
        assertEquals(0, MemoryAllocator.float32ToFp16(0.0f));
    }

    @Test
    @Order(7)
    @DisplayName("FP16: negative one converts correctly")
    void testFp16NegativeOne() {
        short fp16NegOne = MemoryAllocator.float32ToFp16(-1.0f);
        float back = MemoryAllocator.fp16ToFloat32(fp16NegOne);
        assertEquals(-1.0f, back, 0.001f);
    }

    // ── MemoryAllocator Tests ─────────────────────────────────────────────────

    @Test
    @Order(8)
    @DisplayName("MemoryAllocator: INT32 allocation and read/write")
    void testMemoryAllocatorInt32() {
        try (MemoryAllocator alloc = new MemoryAllocator()) {
            int count = 1024;
            MemorySegment seg = alloc.allocateInt32(count, "test-int32");

            assertEquals((long) count * Integer.BYTES, seg.byteSize());

            // Write and read back
            for (int i = 0; i < count; i++) {
                seg.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                        (long) i * Integer.BYTES, i * 7 + 13);
            }
            for (int i = 0; i < count; i++) {
                int val = MemoryAllocator.readInt32(seg, i);
                assertEquals(i * 7 + 13, val, "Read mismatch at index " + i);
            }
        }
    }

    @Test
    @Order(9)
    @DisplayName("MemoryAllocator: FP32 allocation and read/write")
    void testMemoryAllocatorFloat32() {
        try (MemoryAllocator alloc = new MemoryAllocator()) {
            int count = 512;
            MemorySegment seg = alloc.allocateFloat32(count, "test-fp32");

            float[] expected = new float[count];
            for (int i = 0; i < count; i++) {
                expected[i] = i * 0.1f - 25.0f;
                MemoryAllocator.writeFloat32(seg, i, expected[i]);
            }
            for (int i = 0; i < count; i++) {
                assertEquals(expected[i], MemoryAllocator.readFloat32(seg, i), 1e-6f);
            }
        }
    }

    @Test
    @Order(10)
    @DisplayName("MemoryAllocator: copy from array → segment")
    void testCopyFromArray() {
        try (MemoryAllocator alloc = new MemoryAllocator()) {
            int[] src = { 10, 20, 30, 40, 50 };
            MemorySegment seg = alloc.fromInt32Array(src, "test-copy");

            assertEquals((long) src.length * Integer.BYTES, seg.byteSize());
            for (int i = 0; i < src.length; i++) {
                assertEquals(src[i], MemoryAllocator.readInt32(seg, i));
            }
        }
    }

    // ── Dequantization Tests ───────────────────────────────────────────────────

    @Test
    @Order(11)
    @DisplayName("VectorDequantizer: 4-bit synthetic dequantization")
    void testDequantize4BitSynthetic() {
        // Minimal example: 1 input feature, 8 output features, 1 group
        // pack_factor = 8, so 1 INT32 packs all 8 output weights
        GPTQConfig cfg = new GPTQConfig(4, 128, false, false, false, "float32");
        VectorDequantizer deq = new VectorDequantizer(cfg);

        int inF = 1;
        int outF = 8;

        // Pack 8 × 4-bit values: [0,1,2,3,4,5,6,7] into one INT32
        int qweightPacked = 0;
        for (int j = 0; j < 8; j++) {
            qweightPacked |= (j & 0xF) << (j * 4);
        }
        int[] qweight = { qweightPacked }; // shape [1, 1]

        // Zeros: pack 8 × 4-bit value=1 (auto-gptq adds 1 during unpack)
        // After unpack: zeros[j] = (packed_zero & mask) + 1 = 0+1 = 1
        int qzerosPacked = 0; // all zeros packed → after unpack: zero = 0+1 = 1
        int[] qzeros = { qzerosPacked }; // shape [1, 1]

        // Scales: 8 × FP16 = 1.0 each
        short[] scales = new short[8];
        for (int j = 0; j < 8; j++) {
            scales[j] = MemoryAllocator.float32ToFp16(1.0f);
        }

        float[] output = new float[outF * inF];
        deq.dequantize(qweight, qzeros, scales, null, inF, outF, output);

        // Expected: (q[j] - zero[j]) * scale[j] = (j - 1) * 1.0
        for (int j = 0; j < outF; j++) {
            float expected = (j - 1.0f); // zero=1 (packed 0 + autoGPTQ offset)
            assertEquals(expected, output[j], 0.1f,
                    "Mismatch at output feature " + j);
        }
    }

    @Test
    @Order(12)
    @DisplayName("VectorDequantizer: zeros unpacking")
    void testUnpackZeros4Bit() {
        GPTQConfig cfg = GPTQConfig.gptq4bit();
        VectorDequantizer deq = new VectorDequantizer(cfg);

        // Pack zeros with value 3 for all 8 positions
        int packed = 0;
        for (int j = 0; j < 8; j++) {
            packed |= (3 & 0xF) << (j * 4);
        }
        int[] qzeros = { packed };
        float[] zeros = deq.unpackZerosToFloat(qzeros, 1, 8);

        // AutoGPTQ convention: zero = (raw & mask) + 1
        for (int j = 0; j < 8; j++) {
            assertEquals(4.0f, zeros[j], 0.001f, "Zero mismatch at " + j);
        }
    }

    @Test
    @Order(13)
    @DisplayName("VectorDequantizer: mat-vec multiply correctness")
    void testMatVecMul() {
        GPTQConfig cfg = GPTQConfig.gptq4bit();
        VectorDequantizer deq = new VectorDequantizer(cfg);

        // 3×4 weight matrix (identity-ish)
        float[] weight = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0
        };
        float[] input = { 2.0f, 3.0f, 4.0f, 5.0f };
        float[] output = new float[3];

        deq.matVecMul(weight, input, output, 3, 4);

        assertEquals(2.0f, output[0], 0.001f, "Row 0: dot([1,0,0,0],[2,3,4,5])=2");
        assertEquals(3.0f, output[1], 0.001f, "Row 1: dot([0,1,0,0],[2,3,4,5])=3");
        assertEquals(4.0f, output[2], 0.001f, "Row 2: dot([0,0,1,0],[2,3,4,5])=4");
    }

    // ── SafetensorHeader Tests ─────────────────────────────────────────────────

    @Test
    @Order(14)
    @DisplayName("SafetensorHeader.TensorInfo: byte size and numElements")
    void testTensorInfo() {
        GPTQSafetensorHeader.TensorInfo info = new GPTQSafetensorHeader.TensorInfo(
                "F16",
                java.util.List.of(128L, 4096L),
                java.util.List.of(0L, 128L * 4096L * 2L));

        assertEquals(128 * 4096, info.numElements());
        assertEquals(128L * 4096 * 2, info.byteSize());
        assertEquals(2, info.bytesPerElement());
        assertFalse(info.isQuantizedWeight());
    }

    @Test
    @Order(15)
    @DisplayName("SafetensorHeader: add and retrieve tensors")
    void testSafetensorHeader() {
        GPTQSafetensorHeader header = new GPTQSafetensorHeader();
        header.addTensor("model.weight",
                new GPTQSafetensorHeader.TensorInfo("I32",
                        java.util.List.of(512L, 4096L),
                        java.util.List.of(0L, 8388608L)));

        assertTrue(header.hasTensor("model.weight"));
        assertFalse(header.hasTensor("model.bias"));
        assertEquals(1, header.tensorCount());
        assertEquals("I32", header.getTensor("model.weight").getDtype());
    }

    // ── Integration: Full Synthetic Round-Trip ────────────────────────────────

    @Test
    @Order(16)
    @DisplayName("Integration: synthetic quantize → dequantize → verify")
    void testSyntheticQuantizeDequantize() {
        // Simulate quantizing a simple weight and recovering it
        GPTQConfig cfg = GPTQConfig.gptq4bit();
        VectorDequantizer deq = new VectorDequantizer(cfg);

        // Original weight (1 input, 8 outputs): [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5,
        // -2.0]
        // scale = 0.25, zero_q = 8 (center of [0,15])

        float scale = 0.25f;
        float zero_f = 8.0f; // zero-point in quant space

        // Quantize: q = round(w / scale + zero_f)
        float[] original = { 2.0f, 2.25f, 2.5f, 2.75f, 1.5f, 1.25f, 1.0f, 0.75f };
        int[] qVals = new int[8];
        for (int j = 0; j < 8; j++) {
            qVals[j] = Math.max(0, Math.min(15, Math.round(original[j] / scale + zero_f)));
        }

        // Pack 8 INT4 values into 1 INT32
        int packed = 0;
        for (int j = 0; j < 8; j++) {
            packed |= (qVals[j] & 0xF) << (j * 4);
        }
        int[] qweight = { packed };

        // Zeros: store (zero_f - 1) = 7 packed, so after unpack: +1 = 8
        int zeroRaw = (int) (zero_f - 1);
        int zerosPacked = 0;
        for (int j = 0; j < 8; j++) {
            zerosPacked |= (zeroRaw & 0xF) << (j * 4);
        }
        int[] qzeros = { zerosPacked };

        // Scales: all same
        short[] scales = new short[8];
        for (int j = 0; j < 8; j++)
            scales[j] = MemoryAllocator.float32ToFp16(scale);

        float[] output = new float[8];
        deq.dequantize(qweight, qzeros, scales, null, 1, 8, output);

        // Verify: reconstructed ≈ original within quantization error (< scale/2)
        float maxError = scale / 2.0f + 0.01f;
        for (int j = 0; j < 8; j++) {
            assertEquals(original[j], output[j], maxError,
                    "Dequant error too large at j=" + j);
        }
    }
}
