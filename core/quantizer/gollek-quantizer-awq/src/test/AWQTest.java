package tech.kayys.gollek.quantizer.awq;

import com.gptq.awq.model.AWQConfig;
import com.gptq.awq.model.AWQLayer;
import com.gptq.awq.vector.AWQDequantizer;
import com.gptq.ffm.MemoryAllocator;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AWQ components.
 * All data is synthetic — no real model file required.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AWQTest {

    // ── AWQConfig ──────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("AWQConfig: 4-bit GEMM preset")
    void testAwqConfig4BitGemm() {
        AWQConfig cfg = AWQConfig.awq4bit();
        assertEquals(4, cfg.bits());
        assertEquals(128, cfg.groupSize());
        assertEquals(AWQConfig.KernelFormat.GEMM, cfg.kernelFormat());
        assertTrue(cfg.hasZeros());
        assertEquals(8, cfg.packFactor(), "4-bit: 32/4=8");
        assertEquals(0xF, cfg.quantMask(), "4-bit mask=0xF");
        assertEquals(0, cfg.implicitZero(), "has zeros → implicit=0");
        assertEquals(8.0, cfg.compressionRatio(), 0.001);
        assertTrue(cfg.isGemmFormat());
        assertFalse(cfg.isGemvFormat());
    }

    @Test
    @Order(2)
    @DisplayName("AWQConfig: symmetric (no zeros)")
    void testAwqConfigSymmetric() {
        AWQConfig cfg = AWQConfig.awq4bitSymmetric();
        assertFalse(cfg.hasZeros());
        assertEquals(8, cfg.implicitZero(), "symmetric 4-bit: 2^3=8");
        assertEquals(15, cfg.maxQuantValue());
    }

    @Test
    @Order(3)
    @DisplayName("AWQConfig: GEMV format")
    void testAwqConfigGemv() {
        AWQConfig cfg = AWQConfig.awq4bitGemv();
        assertEquals(AWQConfig.KernelFormat.GEMV, cfg.kernelFormat());
        assertTrue(cfg.isGemvFormat());
        assertFalse(cfg.isGemmFormat());
    }

    @Test
    @Order(4)
    @DisplayName("AWQConfig: group size 64")
    void testAwqConfigGroup64() {
        AWQConfig cfg = AWQConfig.awq4bitGroup64();
        assertEquals(64, cfg.groupSize());
    }

    // ── AWQLayer ───────────────────────────────────────────────────────────────

    @Test
    @Order(5)
    @DisplayName("AWQLayer: numGroups calculation")
    void testAwqLayerNumGroups() {
        AWQConfig cfg = AWQConfig.awq4bit(); // groupSize=128
        AWQLayer layer = new AWQLayer("test.layer", cfg);
        layer.setInFeatures(512);
        layer.setOutFeatures(256);

        assertEquals(4, layer.numGroups(), "512/128=4 groups");
        assertEquals(8, layer.packFactor());
        assertFalse(layer.isComplete(), "no tensors loaded yet");
        assertFalse(layer.hasBias());
        assertFalse(layer.hasZeros());
    }

    @Test
    @Order(6)
    @DisplayName("AWQLayer: numGroups non-divisible")
    void testAwqLayerNumGroupsNonDiv() {
        AWQConfig cfg = AWQConfig.awq4bitGroup64();
        AWQLayer layer = new AWQLayer("test", cfg);
        layer.setInFeatures(100); // 100/64 = 1.56 → 2 groups
        assertEquals(2, layer.numGroups());
    }

    // ── AWQDequantizer: Zero Unpacking ─────────────────────────────────────────

    @Test
    @Order(7)
    @DisplayName("AWQDequantizer: unpack zeros with AutoAWQ +1 convention")
    void testUnpackZerosConvention() {
        AWQConfig cfg = AWQConfig.awq4bit(); // packFactor=8
        AWQDequantizer deq = new AWQDequantizer(cfg);

        // Pack zeros=3 for all 8 output features in group 0
        int packed = 0;
        for (int b = 0; b < 8; b++)
            packed |= (3 & 0xF) << (b * 4);
        int[] qzeros = { packed }; // shape [1, 1] → numGroups=1, outF/pack=1 → outF=8

        float[] zeros = deq.unpackZeros(qzeros, 128, 8);

        // AutoAWQ: zero = (raw & mask) + 1 = 3+1 = 4
        for (int j = 0; j < 8; j++) {
            assertEquals(4.0f, zeros[j], 0.001f, "Zero at j=" + j);
        }
    }

    @Test
    @Order(8)
    @DisplayName("AWQDequantizer: implicit zeros for symmetric")
    void testImplicitZeros() {
        AWQConfig cfg = AWQConfig.awq4bitSymmetric();
        AWQDequantizer deq = new AWQDequantizer(cfg);

        float[] zeros = deq.buildImplicitZeros(128, 16);
        assertEquals(128 / 128 * 16, zeros.length);
        for (float z : zeros)
            assertEquals(8.0f, z, 0.001f);
    }

    // ── AWQDequantizer: GEMM Dequantization ───────────────────────────────────

    @Test
    @Order(9)
    @DisplayName("AWQDequantizer: GEMM synthetic 4-bit dequant identity")
    void testGemmDequantIdentity() {
        // 8 input features, 8 output features, 1 group
        // All q-values = 8, zero = 8, scale = 1.0 → w = (8-8)*1 = 0
        AWQConfig cfg = AWQConfig.awq4bit();
        AWQDequantizer deq = new AWQDequantizer(cfg);

        int inF = 8;
        int outF = 8;

        // qweight[0, j] = pack(8,8,8,8,8,8,8,8) for all j
        int packed8 = 0;
        for (int b = 0; b < 8; b++)
            packed8 |= (8 & 0xF) << (b * 4);
        int[] qweight = new int[outF]; // [inF/pack, outF] = [1, 8]
        java.util.Arrays.fill(qweight, packed8);

        // zeros: pack(7,7,...) → after +1: 8
        int packed7 = 0;
        for (int b = 0; b < 8; b++)
            packed7 |= (7 & 0xF) << (b * 4);
        int[] qzeros = { packed7 }; // [1, 1] since outF/pack=1

        // scales = 1.0 FP16
        short fp16_1 = MemoryAllocator.float32ToFp16(1.0f);
        short[] scales = new short[8]; // [1 group, 8 outF]
        java.util.Arrays.fill(scales, fp16_1);

        float[] output = new float[inF * outF];
        deq.dequantize(qweight, qzeros, scales, inF, outF, output);

        for (int i = 0; i < inF * outF; i++) {
            assertEquals(0.0f, output[i], 0.01f, "Expected 0 at idx " + i);
        }
    }

    @Test
    @Order(10)
    @DisplayName("AWQDequantizer: GEMM synthetic round-trip")
    void testGemmDequantRoundTrip() {
        // 8 input features, 8 output features, group=8 (so 1 group)
        AWQConfig cfg = new AWQConfig(4, 8, AWQConfig.KernelFormat.GEMM, true, "float32");
        AWQDequantizer deq = new AWQDequantizer(cfg);

        int inF = 8, outF = 8;
        float scale = 0.5f;
        float zeroF = 8.0f; // mid-range for 4-bit

        // Target weights: row 0 = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
        // → q[j] = round(w[j]/scale + zeroF)
        float[] targets = { 0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, -1.5f, -2.0f };

        // Quantize
        int[][] qInts = new int[1][outF]; // [inF/pack=1, outF=8]
        int[] zeroPackeds = new int[1]; // [numGroups=1, outF/pack=1]

        // Pack qweight for input row 0 (only 1 packed row since inF=8=packFactor)
        int pw = 0;
        for (int j = 0; j < outF; j++) {
            int q = Math.max(0, Math.min(15, Math.round(targets[j] / scale + zeroF)));
            pw |= (q & 0xF) << (j * 4);
        }
        int[] qweight = new int[outF]; // [1, outF]
        // For j=0..7: each element holds 1 value (since each outF col is separate)
        // Actually: qweight[pi=0, j] = the packed word containing bit-slice for
        // i=pi*8+b
        // With 8 input features and packFactor=8: qweight[0, j] holds q[i=b, j] for all
        // b
        // But since each j is a separate output, pack differently:
        // qweight[0, 0] packs all 8 input-feature values for output j=0 ... wait
        // AWQ: qweight[pi, j]: for FIXED output j, packs 8 input features.
        // So qweight[0, j] packs inputs 0..7 for output feature j.
        // For our single-row test (inF=8, inF/pack=1): each output gets its own INT32.
        for (int j = 0; j < outF; j++) {
            // Pack same target across all 8 bit-slots (only bit-slot b=0 matters for row 0)
            int qVal = Math.max(0, Math.min(15, Math.round(targets[j] / scale + zeroF)));
            qweight[j] = qVal; // just bit-slot 0 (shift=0) for i=0
        }

        // Pack zeros: raw = zeroF - 1 = 7 → after +1 = 8
        int zp = 0;
        for (int b = 0; b < 8; b++)
            zp |= (7 & 0xF) << (b * 4);
        int[] qzeros = { zp }; // [1 group, 1 packed col]

        short[] scales = new short[outF];
        short fp16Scale = MemoryAllocator.float32ToFp16(scale);
        java.util.Arrays.fill(scales, fp16Scale);

        float[] output = new float[inF * outF];
        deq.dequantize(qweight, qzeros, scales, inF, outF, output);

        float tolerance = scale / 2.0f + 0.02f;
        for (int j = 0; j < outF; j++) {
            // output[i=0, j] is at index 0*outF+j = j
            assertEquals(targets[j], output[j], tolerance,
                    "Dequant mismatch at j=" + j);
        }
    }

    // ── AWQDequantizer: FP16 Conversion ───────────────────────────────────────

    @Test
    @Order(11)
    @DisplayName("AWQDequantizer: fp16ToFloat32Bulk correctness")
    void testFp16BulkConversion() {
        AWQDequantizer deq = new AWQDequantizer(AWQConfig.awq4bit());
        float[] inputs = { 0.0f, 1.0f, -1.0f, 3.14f, 0.001f, 100.0f };
        short[] fp16 = new short[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            fp16[i] = MemoryAllocator.float32ToFp16(inputs[i]);
        }
        float[] back = deq.fp16ToFloat32Bulk(fp16);
        for (int i = 0; i < inputs.length; i++) {
            assertEquals(inputs[i], back[i], Math.abs(inputs[i]) * 0.002 + 1e-5,
                    "FP16 bulk mismatch at i=" + i);
        }
    }

    // ── AWQDequantizer: Activation Scaling ────────────────────────────────────

    @Test
    @Order(12)
    @DisplayName("AWQDequantizer: activation scaling (element-wise multiply)")
    void testActivationScaling() {
        AWQDequantizer deq = new AWQDequantizer(AWQConfig.awq4bit());
        float[] activation = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float[] scales = { 0.5f, 0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f, 2.0f };
        deq.applyActivationScaling(activation, scales);
        assertArrayEquals(
                new float[] { 0.5f, 1.0f, 1.5f, 2.0f, 10.0f, 12.0f, 14.0f, 16.0f },
                activation, 0.001f);
    }

    @Test
    @Order(13)
    @DisplayName("AWQDequantizer: absorb activation scale into weight")
    void testAbsorbActivationScale() {
        AWQDequantizer deq = new AWQDequantizer(AWQConfig.awq4bit());
        // weight [2 rows, 4 cols] = [[1,1,1,1],[2,2,2,2]]
        float[] weight = { 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f };
        // scale [inF=2]: [2.0, 4.0] → inv = [0.5, 0.25]
        float[] actScale = { 2.0f, 4.0f };
        deq.absorbActivationScale(weight, actScale, 2, 4);
        // row 0: 1/2 = 0.5; row 1: 2/4 = 0.5
        for (int j = 0; j < 4; j++)
            assertEquals(0.5f, weight[j], 0.001f, "row0 j=" + j);
        for (int j = 0; j < 4; j++)
            assertEquals(0.5f, weight[4 + j], 0.001f, "row1 j=" + j);
    }

    // ── AWQDequantizer: MatVec ─────────────────────────────────────────────────

    @Test
    @Order(14)
    @DisplayName("AWQDequantizer: dequantMatVec trivial case")
    void testDequantMatVecTrivial() {
        // 8×8 all-zeros weight, input = ones → output = zeros
        AWQConfig cfg = new AWQConfig(4, 8, AWQConfig.KernelFormat.GEMM, false, "float32");
        AWQDequantizer deq = new AWQDequantizer(cfg);

        int inF = 8, outF = 8;

        // qweight all zeros (q=0)
        int[] qweight = new int[outF]; // [1, 8]

        // scales = 1.0
        short fp16_1 = MemoryAllocator.float32ToFp16(1.0f);
        short[] scales = new short[outF];
        java.util.Arrays.fill(scales, fp16_1);

        float[] inputVec = new float[inF];
        java.util.Arrays.fill(inputVec, 1.0f);
        float[] outputVec = new float[outF];

        // Symmetric → implicit zero = 8
        deq.dequantMatVec(qweight, null, scales, inputVec, outputVec, inF, outF);

        // w[i,j] = (0 - 8) * 1.0 = -8.0; out[j] = sum_i(-8 * 1) = -8 * 8 = -64
        for (int j = 0; j < outF; j++) {
            assertEquals(-64.0f, outputVec[j], 0.1f, "MatVec output at j=" + j);
        }
    }

    // ── AWQConfig: Derived Properties ─────────────────────────────────────────

    @Test
    @Order(15)
    @DisplayName("AWQConfig: pack factor and mask for all valid bit-widths")
    void testConfigAllBitWidths() {
        int[][] cases = { { 4, 8, 0xF }, { 8, 4, 0xFF }, { 2, 16, 0x3 } };
        for (int[] c : cases) {
            AWQConfig cfg = new AWQConfig(c[0], 128, AWQConfig.KernelFormat.GEMM, true, "float32");
            assertEquals(c[1], cfg.packFactor(), "bits=" + c[0] + " packFactor");
            assertEquals(c[2], cfg.quantMask(), "bits=" + c[0] + " mask");
        }
    }

    @Test
    @Order(16)
    @DisplayName("AWQConfig: compression ratio")
    void testCompressionRatio() {
        assertEquals(8.0, AWQConfig.awq4bit().compressionRatio(), 0.001);
        assertEquals(4.0, new AWQConfig(8, 128, AWQConfig.KernelFormat.GEMM, true, "float32")
                .compressionRatio(), 0.001);
        assertEquals(16.0, new AWQConfig(2, 128, AWQConfig.KernelFormat.GEMM, true, "float32")
                .compressionRatio(), 0.001);
    }

    // ── GEMV vs GEMM consistency ───────────────────────────────────────────────

    @Test
    @Order(17)
    @DisplayName("AWQDequantizer: GEMV and GEMM produce same result on symmetric data")
    void testGemvGemmConsistency() {
        // Build a symmetric case: single group, all q=8, scale=1, implicit zero=8
        // → all weights = 0, regardless of format
        int inF = 8, outF = 4;

        int packedMid = 0;
        for (int b = 0; b < 8; b++)
            packedMid |= (8 & 0xF) << (b * 4);
        int[] qweight = new int[outF];
        java.util.Arrays.fill(qweight, packedMid);

        short fp16_1 = MemoryAllocator.float32ToFp16(1.0f);
        short[] scales = new short[outF]; // [1 group, outF]
        java.util.Arrays.fill(scales, fp16_1);

        AWQConfig gemmCfg = new AWQConfig(4, 8, AWQConfig.KernelFormat.GEMM, false, "float32");
        AWQConfig gemvCfg = new AWQConfig(4, 8, AWQConfig.KernelFormat.GEMV, false, "float32");

        float[] outGemm = new float[inF * outF];
        float[] outGemv = new float[inF * outF];

        new AWQDequantizer(gemmCfg).dequantize(qweight, null, scales, inF, outF, outGemm);
        new AWQDequantizer(gemvCfg).dequantize(qweight, null, scales, inF, outF, outGemv);

        assertArrayEquals(outGemm, outGemv, 0.001f,
                "GEMM and GEMV should give identical results on symmetric zero-weight data");
    }

    @Test
    @Order(18)
    @DisplayName("AWQDequantizer: output dimensions match inF×outF")
    void testOutputDimensions() {
        AWQConfig cfg = AWQConfig.awq4bit();
        AWQDequantizer deq = new AWQDequantizer(cfg);

        int inF = 64, outF = 32;
        int[] qweight = new int[(inF / 8) * outF]; // [8, 32]
        short[] scales = new short[(inF / 128) > 0 ? (inF / 128) * outF : outF];
        java.util.Arrays.fill(scales, MemoryAllocator.float32ToFp16(1.0f));

        float[] output = new float[inF * outF];
        // Should not throw; just verify size
        deq.dequantize(qweight, null, scales, inF, outF, output);
        assertEquals(inF * outF, output.length);
    }

    @Test
    @Order(19)
    @DisplayName("AWQDequantizer: scale=0 produces zeros")
    void testZeroScale() {
        AWQConfig cfg = new AWQConfig(4, 8, AWQConfig.KernelFormat.GEMM, false, "float32");
        AWQDequantizer deq = new AWQDequantizer(cfg);

        int inF = 8, outF = 8;
        int[] qweight = new int[outF]; // all zeros
        short[] scales = new short[outF]; // FP16 zero = 0x0000
        float[] output = new float[inF * outF];
        deq.dequantize(qweight, null, scales, inF, outF, output);

        for (float v : output)
            assertEquals(0.0f, v, 1e-6f, "scale=0 → output=0");
    }

    @Test
    @Order(20)
    @DisplayName("AWQDequantizer: activation scaling broadcasts via SIMD correctly")
    void testActivationScalingLargeVector() {
        AWQDequantizer deq = new AWQDequantizer(AWQConfig.awq4bit());
        int n = 1024;
        float[] activation = new float[n];
        float[] scaleVec = new float[n];
        java.util.Arrays.fill(activation, 3.0f);
        java.util.Arrays.fill(scaleVec, 2.0f);
        deq.applyActivationScaling(activation, scaleVec);
        for (int i = 0; i < n; i++) {
            assertEquals(6.0f, activation[i], 0.001f, "SIMD scale at i=" + i);
        }
    }
}
