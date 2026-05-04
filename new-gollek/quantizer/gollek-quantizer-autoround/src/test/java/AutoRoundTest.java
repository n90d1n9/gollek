package tech.kayys.gollek.quantizer.autoround;

import tech.kayys.gollek.quantizer.autoround.AutoRoundConfig;
import tech.kayys.gollek.quantizer.autoround.AutoRoundLayer;
import tech.kayys.gollek.quantizer.autoround.AutoRoundDequantizer;
import tech.kayys.gollek.quantizer.gptq.MemoryAllocator;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AutoRound components.
 * All data is synthetic — no real model file required.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AutoRoundTest {

    // ── AutoRoundConfig ────────────────────────────────────────────────────────

    @Test @Order(1)
    @DisplayName("AutoRoundConfig: native 4-bit presets")
    void testConfig4bitNative() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bit();
        assertEquals(4,   cfg.bits());
        assertEquals(128, cfg.groupSize());
        assertTrue(cfg.hasZeroPoint());
        assertEquals(AutoRoundConfig.ScaleDtype.FLOAT32, cfg.scaleDtype());
        assertEquals(AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, cfg.packFormat());
        assertEquals(8,    cfg.packFactor());
        assertEquals(0xF,  cfg.quantMask());
        assertEquals(0,    cfg.implicitZeroPoint(), "has zp → implicit = 0");
        assertEquals(8.0,  cfg.compressionRatio(), 0.001);
        assertTrue(cfg.isNativeFormat());
        assertFalse(cfg.isGptqCompatFormat());
        assertFalse(cfg.isItrexFormat());
    }

    @Test @Order(2)
    @DisplayName("AutoRoundConfig: symmetric (no zero-point)")
    void testConfigSymmetric() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitSymmetric();
        assertFalse(cfg.hasZeroPoint());
        assertEquals(8, cfg.implicitZeroPoint(), "4-bit symmetric: 2^3=8");
        assertFalse(cfg.hasFp16Scales());
        assertTrue(cfg.hasFp32Scales());
    }

    @Test @Order(3)
    @DisplayName("AutoRoundConfig: GPTQ-compat has FP16 scales")
    void testConfigGptqCompat() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRoundGptqCompat();
        assertTrue(cfg.isGptqCompatFormat());
        assertTrue(cfg.hasFp16Scales());
        assertFalse(cfg.hasFp32Scales());
        assertEquals(AutoRoundConfig.ScaleDtype.FLOAT16, cfg.scaleDtype());
    }

    @Test @Order(4)
    @DisplayName("AutoRoundConfig: pack factor for all bit-widths")
    void testPackFactorAll() {
        assertEquals(8,  AutoRoundConfig.autoRound4bit().packFactor());
        assertEquals(4,  AutoRoundConfig.autoRound8bit().packFactor());
        assertEquals(16, AutoRoundConfig.autoRound2bit().packFactor());
    }

    @Test @Order(5)
    @DisplayName("AutoRoundConfig: group size 32")
    void testGroup32() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitGroup32();
        assertEquals(32, cfg.groupSize());
    }

    // ── AutoRoundLayer ─────────────────────────────────────────────────────────

    @Test @Order(6)
    @DisplayName("AutoRoundLayer: numGroups and packFactor")
    void testLayerDerived() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bit();
        AutoRoundLayer layer = new AutoRoundLayer("test.layer", cfg);
        layer.setInFeatures(512);
        layer.setOutFeatures(256);
        assertEquals(4, layer.numGroups(), "512/128=4");
        assertEquals(8, layer.packFactor());
        assertFalse(layer.isComplete());
    }

    @Test @Order(7)
    @DisplayName("AutoRoundLayer: numGroups non-divisible input")
    void testLayerGroupsNonDiv() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitGroup32();
        AutoRoundLayer layer = new AutoRoundLayer("x", cfg);
        layer.setInFeatures(100);  // 100/32 → 4 groups (ceil)
        assertEquals(4, layer.numGroups());
    }

    // ── AutoRoundDequantizer: BF16 Conversion ─────────────────────────────────

    @Test @Order(8)
    @DisplayName("AutoRoundDequantizer: bf16ToFloat32 round-trip")
    void testBf16ToFp32() {
        // BF16 = top 16 bits of FP32 — should recover exact values that fit in 7 mantissa bits
        float[] vals = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 100.0f};
        for (float v : vals) {
            short bf16  = (short) (Float.floatToIntBits(v) >>> 16);
            float back  = AutoRoundDequantizer.bf16ToFloat32(bf16);
            // BF16 has ~2-3 significant decimal digits
            assertEquals(v, back, Math.abs(v) * 0.01 + 1e-5,
                "BF16 round-trip failed for: " + v);
        }
    }

    @Test @Order(9)
    @DisplayName("AutoRoundDequantizer: bf16 bulk array conversion")
    void testBf16BulkArray() {
        float[] inputs = {1.0f, 2.0f, -3.0f, 0.25f};
        short[] bf16   = new short[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            bf16[i] = (short) (Float.floatToIntBits(inputs[i]) >>> 16);
        }
        float[] back = AutoRoundDequantizer.bf16ArrayToFloat32(bf16);
        for (int i = 0; i < inputs.length; i++) {
            assertEquals(inputs[i], back[i], Math.abs(inputs[i]) * 0.01 + 1e-5);
        }
    }

    // ── AutoRoundDequantizer: Zero-Point Handling ──────────────────────────────

    @Test @Order(10)
    @DisplayName("AutoRoundDequantizer: buildZeroArray symmetric (implicit zp=8)")
    void testBuildZeroSymmetric() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitSymmetric(); // no zp
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);
        // inF=128 → 1 group; outF=16
        float[] zeros = dq.buildZeroArray(null, 16, 128);
        assertEquals(16, zeros.length); // outF × numGroups = 16×1
        for (float z : zeros) assertEquals(8.0f, z, 0.001f);
    }

    @Test @Order(11)
    @DisplayName("AutoRoundDequantizer: buildZeroArray from plain INT32")
    void testBuildZeroFromPlainInt() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bit();
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);
        // outF=4, inF=128 → 1 group, zpInts[outF×numGroups] = [3, 5, 7, 9]
        int[] zpInts = {3, 5, 7, 9};
        float[] zeros = dq.buildZeroArray(zpInts, 4, 128);
        assertArrayEquals(new float[]{3f, 5f, 7f, 9f}, zeros, 0.001f);
    }

    @Test @Order(12)
    @DisplayName("AutoRoundDequantizer: unpackGptqCompatZeros (+1 bias)")
    void testUnpackGptqCompatZeros() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bit(); // packFactor=8
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        // Pack zeros: 8 output features, raw value=5
        // After +1 bias: zp = 6
        int packed = 0;
        for (int b = 0; b < 8; b++) packed |= (5 & 0xF) << (b * 4);
        // qzeros shape [numGroups=1, outF/pack=1]
        int[] qzeros = {packed};
        // unpack to plain [outF=8, numGroups=1]
        int[] plain = dq.unpackGptqCompatZeros(qzeros, 8, 1);

        assertEquals(8, plain.length);
        for (int j = 0; j < 8; j++) {
            assertEquals(6, plain[j], "GPTQ-compat ZP should be 5+1=6 at j=" + j);
        }
    }

    // ── AutoRoundDequantizer: Scale Transpose ──────────────────────────────────

    @Test @Order(13)
    @DisplayName("AutoRoundDequantizer: transposeFp16ScalesToFp32")
    void testTransposeFp16Scales() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRoundGptqCompat();
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        // GPTQ layout: scales[g, j] — 2 groups, 4 output features
        // g=0: scales = [1.0, 2.0, 3.0, 4.0]
        // g=1: scales = [5.0, 6.0, 7.0, 8.0]
        int numGroups = 2, outF = 4;
        short[] fp16Scales = new short[numGroups * outF];
        float[] expected   = new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        for (int i = 0; i < expected.length; i++) {
            fp16Scales[i] = MemoryAllocator.float32ToFp16(expected[i]);
        }

        float[] fp32 = dq.transposeFp16ScalesToFp32(fp16Scales, outF, numGroups);

        // AutoRound layout: scales[j, g]
        // j=0: scale[0,0]=1.0, scale[0,1]=5.0
        // j=1: scale[1,0]=2.0, scale[1,1]=6.0  etc.
        assertEquals(1.0f, fp32[0 * numGroups + 0], 0.01f, "j=0,g=0");
        assertEquals(5.0f, fp32[0 * numGroups + 1], 0.01f, "j=0,g=1");
        assertEquals(2.0f, fp32[1 * numGroups + 0], 0.01f, "j=1,g=0");
        assertEquals(6.0f, fp32[1 * numGroups + 1], 0.01f, "j=1,g=1");
    }

    // ── AutoRoundDequantizer: Native Dequantization ────────────────────────────

    @Test @Order(14)
    @DisplayName("AutoRoundDequantizer: native identity — q=zp → w=0")
    void testNativeDequantIdentityZero() {
        // All q = zp → w = 0 for any scale
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bit();
        cfg = new AutoRoundConfig(4, 8, true, AutoRoundConfig.ScaleDtype.FLOAT32,
            AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, null, null);
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 8, outF = 8;
        int packFactor = 8;  // 4-bit: 32/4=8

        // qweight[outF/8=1, inF=8]: pack zp=5 for all positions
        int packed5 = 0;
        for (int b = 0; b < 8; b++) packed5 |= (5 & 0xF) << (b * 4);
        int[] qweight = new int[inF]; // [1 packed_row, 8 cols]
        java.util.Arrays.fill(qweight, packed5);

        // scale [outF=8, numGroups=1] = 1.0
        float[] scales = new float[outF]; // outF × numGroups = 8 × 1
        java.util.Arrays.fill(scales, 1.0f);

        // zp [outF=8, numGroups=1] = 5
        int[] zp = new int[outF];
        java.util.Arrays.fill(zp, 5);

        float[] output = new float[outF * inF];
        dq.dequantize(qweight, scales, zp, inF, outF, output);

        for (int k = 0; k < output.length; k++) {
            assertEquals(0.0f, output[k], 0.01f, "q=zp → w=0 at k=" + k);
        }
    }

    @Test @Order(15)
    @DisplayName("AutoRoundDequantizer: native round-trip quantize → dequantize")
    void testNativeRoundTrip() {
        // 8 output features, 8 input features, group=8 (1 group)
        AutoRoundConfig cfg = new AutoRoundConfig(4, 8, true, AutoRoundConfig.ScaleDtype.FLOAT32, AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, null, null);
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 8, outF = 8;
        float scale = 0.25f;
        int   zp    = 8;           // mid-range 4-bit

        // Original weights for output row j=0: ramp from -1.75 to 0.0
        float[] targets = {-1.75f, -1.5f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, 0.0f};

        // Quantize: q[j=0, i] = round(targets[i] / scale) + zp
        // qweight[pr=0, i] packs all 8 output-feature values for input i
        // But outF=8, so packed row 0 holds all 8 output features for col i:
        // For col i, the packed INT32 holds q[j=0..7, i]

        int[] qweight = new int[inF]; // [outF/pack=1, inF=8]
        for (int i = 0; i < inF; i++) {
            // Only output row j=0 is non-trivial (others use q=zp=8 → w=0)
            int qVal0 = Math.max(0, Math.min(15, Math.round(targets[i] / scale) + zp));
            // Pack: bit-slice b=0 (j=0) = qVal0; b=1..7 (j=1..7) = zp
            int packed = qVal0; // bit 0..3
            for (int b = 1; b < 8; b++) packed |= (zp & 0xF) << (b * 4);
            qweight[i] = packed;
        }

        // Scale [outF=8, 1 group]: all 1.0 except j=0 which uses 0.25
        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 1.0f);
        scales[0] = scale;

        // ZP [outF=8, 1 group]: all = zp
        int[] zpArr = new int[outF];
        java.util.Arrays.fill(zpArr, zp);

        float[] output = new float[outF * inF];
        dq.dequantize(qweight, scales, zpArr, inF, outF, output);

        float tol = scale / 2.0f + 0.01f;
        for (int i = 0; i < inF; i++) {
            // output row j=0 is output[j=0, i] = output[0*inF + i]
            assertEquals(targets[i], output[0 * inF + i], tol,
                "Round-trip mismatch at (j=0, i=" + i + ")");
        }
    }

    @Test @Order(16)
    @DisplayName("AutoRoundDequantizer: symmetric config uses implicit ZP correctly")
    void testSymmetricImplicitZP() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitSymmetric(); // no ZP stored
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 8, outF = 8;
        // q = 8 (= implicit zp) everywhere → w = 0
        int packed8 = 0;
        for (int b = 0; b < 8; b++) packed8 |= (8 & 0xF) << (b * 4);
        int[] qweight = new int[inF];
        java.util.Arrays.fill(qweight, packed8);

        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 2.0f);

        float[] output = new float[outF * inF];
        dq.dequantize(qweight, scales, null, inF, outF, output);  // null zp

        for (float v : output) assertEquals(0.0f, v, 0.01f);
    }

    // ── AutoRoundDequantizer: Fused MatVec ─────────────────────────────────────

    @Test @Order(17)
    @DisplayName("AutoRoundDequantizer: dequantMatVec matches separate dequant+dot")
    void testDequantMatVecConsistency() {
        AutoRoundConfig cfg = new AutoRoundConfig(4, 8,
            false, AutoRoundConfig.ScaleDtype.FLOAT32,
            AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, null, null);
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 8, outF = 8;
        // All q=4, scale=0.5, implicit zp=8 → w = (4-8)*0.5 = -2.0 everywhere
        int packed4 = 0;
        for (int b = 0; b < 8; b++) packed4 |= (4 & 0xF) << (b * 4);
        int[] qweight = new int[inF];
        java.util.Arrays.fill(qweight, packed4);

        float[] scales = new float[outF];
        java.util.Arrays.fill(scales, 0.5f);

        float[] inputVec  = new float[inF];
        java.util.Arrays.fill(inputVec, 1.0f);
        float[] outputVec = new float[outF];

        dq.dequantMatVec(qweight, scales, null, inputVec, outputVec, inF, outF);

        // Expected: dot(-2, [1,1,1,1,1,1,1,1]) = -16.0 per output feature
        for (int j = 0; j < outF; j++) {
            assertEquals(-16.0f, outputVec[j], 0.1f, "MatVec at j=" + j);
        }
    }

    @Test @Order(18)
    @DisplayName("AutoRoundDequantizer: batched matmul output dimensions")
    void testBatchedMatMulDimensions() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitSymmetric();
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 16, outF = 8, batchSize = 3;
        // qweight [outF/pack=1, inF=16]: all packed q=8 → w=0
        int packed8 = 0;
        for (int b = 0; b < 8; b++) packed8 |= (8 & 0xF) << (b * 4);
        int[] qweight = new int[outF / 8 * inF]; // [1, 16]
        java.util.Arrays.fill(qweight, packed8);

        float[] scales  = new float[outF];
        java.util.Arrays.fill(scales, 1.0f);

        float[] inputs  = new float[batchSize * inF];
        java.util.Arrays.fill(inputs, 1.0f);
        float[] outputs = new float[batchSize * outF];

        dq.dequantMatMul(qweight, scales, null, inputs, outputs, inF, outF, batchSize);

        // w=0 everywhere → all outputs should be 0
        for (int k = 0; k < outputs.length; k++) {
            assertEquals(0.0f, outputs[k], 0.01f, "Batched matmul output at k=" + k);
        }
    }

    // ── AutoRoundConfig: toString ──────────────────────────────────────────────

    @Test @Order(19)
    @DisplayName("AutoRoundConfig: toString contains all key fields")
    void testConfigToString() {
        String s = AutoRoundConfig.autoRound4bit().toString();
        assertTrue(s.contains("bits=4"));
        assertTrue(s.contains("groupSize=128"));
        assertTrue(s.contains("hasZP=true"));
        assertTrue(s.contains("FLOAT32"));
        assertTrue(s.contains("AUTOROUND_NATIVE"));
    }

    // ── AutoRoundLayer: estimatedBytes ────────────────────────────────────────

    @Test @Order(20)
    @DisplayName("AutoRoundLayer: estimatedBytes with no segments = 0")
    void testLayerEstimatedBytesEmpty() {
        AutoRoundLayer layer = new AutoRoundLayer("empty", AutoRoundConfig.autoRound4bit());
        assertEquals(0L, layer.estimatedBytes());
        assertFalse(layer.isComplete());
        assertFalse(layer.hasBias());
        assertFalse(layer.hasZp());
    }

    // ── Edge Cases ────────────────────────────────────────────────────────────

    @Test @Order(21)
    @DisplayName("AutoRoundDequantizer: output[outF × inF] has correct element count")
    void testOutputElementCount() {
        AutoRoundConfig cfg = AutoRoundConfig.autoRound4bitSymmetric();
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 64, outF = 32;
        int[] qweight = new int[outF / 8 * inF]; // [4, 64]
        float[] scales = new float[outF * 1];     // [32, 1]
        java.util.Arrays.fill(scales, 1.0f);

        float[] output = new float[outF * inF];
        dq.dequantize(qweight, scales, null, inF, outF, output);
        assertEquals(outF * inF, output.length);
    }

    @Test @Order(22)
    @DisplayName("AutoRoundDequantizer: scale=2x doubles output values")
    void testScaleDoubles() {
        AutoRoundConfig cfg = new AutoRoundConfig(4, 8,
            false, AutoRoundConfig.ScaleDtype.FLOAT32,
            AutoRoundConfig.PackFormat.AUTOROUND_NATIVE, "float32", 200, 0.001, false, 128, 2048, null, null);
        AutoRoundDequantizer dq = new AutoRoundDequantizer(cfg);

        int inF = 8, outF = 8;
        // q=10 everywhere, implicit zp=8, scale=1 or 2
        int packed10 = 0;
        for (int b = 0; b < 8; b++) packed10 |= (10 & 0xF) << (b * 4);
        int[] qweight = new int[inF];
        java.util.Arrays.fill(qweight, packed10);

        float[] scales1 = new float[outF]; java.util.Arrays.fill(scales1, 1.0f);
        float[] scales2 = new float[outF]; java.util.Arrays.fill(scales2, 2.0f);

        float[] out1 = new float[outF * inF];
        float[] out2 = new float[outF * inF];

        dq.dequantize(qweight, scales1, null, inF, outF, out1);
        dq.dequantize(qweight, scales2, null, inF, outF, out2);

        // w = (10-8)*scale → scale=1: w=2, scale=2: w=4
        for (int k = 0; k < out1.length; k++) {
            assertEquals(2.0f, out1[k], 0.01f, "scale=1 at k=" + k);
            assertEquals(4.0f, out2[k], 0.01f, "scale=2 at k=" + k);
        }
    }
}
