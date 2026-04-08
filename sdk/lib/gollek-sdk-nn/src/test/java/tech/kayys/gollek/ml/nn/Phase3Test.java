package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.models.BERT;
import tech.kayys.gollek.ml.models.ResNet;
import tech.kayys.gollek.ml.optimize.FP16Quantizer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 3 smoke tests: FP16 quantizer, TransformerBlock, TransformerEncoder,
 * ResNet model zoo, BERT model zoo.
 */
class Phase3Test {

    // ── FP16Quantizer ─────────────────────────────────────────────────────

    @Test
    void fp16RoundTripNormals() {
        FP16Quantizer q = new FP16Quantizer();
        GradTensor t = GradTensor.of(new float[]{1.0f, -1.0f, 0.5f, 3.14f, -2.71f}, 5);
        FP16Quantizer.FP16Tensor fp16 = q.quantize(t);
        GradTensor restored = q.dequantize(fp16);

        float[] orig = t.data(), rec = restored.data();
        for (int i = 0; i < orig.length; i++)
            assertEquals(orig[i], rec[i], 0.001f, "FP16 round-trip error at index " + i);
    }

    @Test
    void fp16SpecialValues() {
        assertEquals(0f,              FP16Quantizer.fp16ToFloat(FP16Quantizer.floatToFp16(0f)),      1e-6f);
        assertEquals(Float.POSITIVE_INFINITY,
            FP16Quantizer.fp16ToFloat(FP16Quantizer.floatToFp16(Float.POSITIVE_INFINITY)), 1e-6f);
        assertTrue(Float.isNaN(FP16Quantizer.fp16ToFloat(FP16Quantizer.floatToFp16(Float.NaN))));
    }

    @Test
    void fp16CompressionRatio() {
        assertEquals(2.0f, new FP16Quantizer().compressionRatio(), 1e-5f);
    }

    @Test
    void fp16PreservesShape() {
        FP16Quantizer q = new FP16Quantizer();
        GradTensor t = GradTensor.randn(4, 8);
        assertArrayEquals(new long[]{4, 8}, q.quantize(t).shape());
    }

    // ── TransformerBlock ──────────────────────────────────────────────────

    @Test
    void transformerBlockOutputShape() {
        TransformerBlock block = new TransformerBlock(64, 4, 256, 0.0f);
        GradTensor x = GradTensor.randn(2, 10, 64); // [B=2, T=10, dModel=64]
        GradTensor out = block.forward(x);
        assertArrayEquals(new long[]{2, 10, 64}, out.shape());
    }

    @Test
    void transformerBlockEvalMode() {
        TransformerBlock block = new TransformerBlock(32, 2, 128, 0.5f);
        block.eval();
        GradTensor x = GradTensor.randn(1, 5, 32);
        // Should not throw; dropout is identity in eval mode
        assertDoesNotThrow(() -> block.forward(x));
    }

    // ── TransformerEncoder ────────────────────────────────────────────────

    @Test
    void transformerEncoderOutputShape() {
        TransformerEncoder enc = TransformerEncoder.builder()
            .dModel(64).nHeads(4).dFF(256).nLayers(2).maxSeqLen(32).dropout(0f)
            .build();
        GradTensor x = GradTensor.randn(2, 8, 64);
        GradTensor out = enc.forward(x);
        assertArrayEquals(new long[]{2, 8, 64}, out.shape());
    }

    @Test
    void transformerEncoderNumLayers() {
        TransformerEncoder enc = TransformerEncoder.builder()
            .dModel(32).nHeads(2).dFF(64).nLayers(4).build();
        assertEquals(4, enc.numLayers());
    }

    @Test
    void transformerEncoderHasParameters() {
        TransformerEncoder enc = TransformerEncoder.builder()
            .dModel(64).nHeads(4).dFF(256).nLayers(2).build();
        assertTrue(enc.parameterCount() > 0);
    }

    // ── ResNet ────────────────────────────────────────────────────────────

    @Test
    void resnet18OutputShape() {
        NNModule model = ResNet.resnet18(10);
        // Use small spatial input to keep test fast
        GradTensor x = GradTensor.randn(2, 3, 32, 32);
        GradTensor out = model.forward(x);
        assertArrayEquals(new long[]{2, 10}, out.shape());
    }

    @Test
    void resnet18HasParameters() {
        NNModule model = ResNet.resnet18(1000);
        assertTrue(model.parameterCount() > 1_000_000L,
            "ResNet-18 should have >1M parameters");
    }

    @Test
    void resnet50OutputShape() {
        NNModule model = ResNet.resnet50(10);
        GradTensor x = GradTensor.randn(1, 3, 32, 32);
        GradTensor out = model.forward(x);
        assertArrayEquals(new long[]{1, 10}, out.shape());
    }

    // ── BERT ──────────────────────────────────────────────────────────────

    @Test
    void bertBaseOutputShape() {
        // Use tiny vocab/dim for speed
        TransformerEncoder bert = TransformerEncoder.builder()
            .dModel(64).nHeads(4).dFF(256).nLayers(2).maxSeqLen(16).dropout(0f)
            .build();
        GradTensor x = GradTensor.randn(2, 8, 64);
        GradTensor out = bert.forward(x);
        assertArrayEquals(new long[]{2, 8, 64}, out.shape());
    }

    @Test
    void bertForClassificationOutputShape() {
        // Tiny BERT for classification
        TransformerEncoder enc = TransformerEncoder.builder()
            .dModel(768).nHeads(12).dFF(3072).nLayers(1).maxSeqLen(16).dropout(0f)
            .build();
        BERT.BertForClassification model = new BERT.BertForClassification(enc, 2);
        GradTensor x = GradTensor.randn(2, 8, 768);
        GradTensor out = model.forward(x);
        assertArrayEquals(new long[]{2, 2}, out.shape());
    }
}
