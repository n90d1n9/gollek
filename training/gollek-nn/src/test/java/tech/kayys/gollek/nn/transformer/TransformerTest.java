package tech.kayys.gollek.ml.transformer;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.layer.Dropout;
import tech.kayys.gollek.ml.nn.layer.GELU;

import static org.junit.jupiter.api.Assertions.*;

class TransformerTest {

    @Test
    void testMultiHeadAttentionCreation() {
        var mha = new MultiHeadAttention(768, 12);
        assertEquals(768, mha.getEmbedDim());
        assertEquals(12, mha.getNumHeads());
    }

    @Test
    void testMultiHeadAttentionForward() {
        var mha = new MultiHeadAttention(768, 12);
        var input = GradTensor.randn(2, 10, 768); // [batch, seq, dim]
        var output = mha.forward(input);
        assertArrayEquals(new long[]{2, 10, 768}, output.shape());
    }

    @Test
    void multiHeadAttentionBackpropagatesThroughHeadReshapes() {
        var mha = new MultiHeadAttention(2, 1);
        setIdentityProjectionWeights(mha);
        float[] values = new float[] {0.2f, -0.1f, 0.4f, 0.3f};
        var input = GradTensor.of(values, 1, 2, 2).requiresGrad(true);

        mha.forward(input).sum().backward();

        assertNotNull(input.grad());
        assertNotNull(mha.namedParameters().get("q_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("k_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("v_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("out_proj.weight").grad());
        assertArrayEquals(
                finiteDifferenceInputGradient(mha, values, false),
                input.grad().data(),
                3e-3f);
    }

    @Test
    void multiHeadAttentionCausalMaskBackpropagatesThroughMaskApplication() {
        var mha = new MultiHeadAttention(2, 1);
        setIdentityProjectionWeights(mha);
        float[] values = new float[] {0.2f, -0.1f, 0.4f, 0.3f};
        var input = GradTensor.of(values, 1, 2, 2).requiresGrad(true);

        mha.forward(input, true).sum().backward();

        assertNotNull(input.grad());
        assertNotNull(mha.namedParameters().get("q_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("k_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("v_proj.weight").grad());
        assertNotNull(mha.namedParameters().get("out_proj.weight").grad());
        assertArrayEquals(
                finiteDifferenceInputGradient(mha, values, true),
                input.grad().data(),
                3e-3f);
    }

    @Test
    void multiHeadAttentionAppliesConstructorDropoutDuringTraining() {
        var mha = new MultiHeadAttention(2, 1, 1f);
        setIdentityProjectionWeights(mha);

        GradTensor output = mha.forward(GradTensor.of(new float[] {
                0.2f, -0.1f,
                0.4f, 0.3f
        }, 1, 2, 2));

        assertArrayEquals(new float[] {0f, 0f, 0f, 0f}, output.data(), 1e-6f);
    }

    @Test
    void multiHeadAttentionDisablesAttentionDropoutDuringEval() {
        var dropoutAttention = new MultiHeadAttention(2, 1, 1f);
        var baseline = new MultiHeadAttention(2, 1, 0f);
        dropoutAttention.eval();
        baseline.eval();
        setIdentityProjectionWeights(dropoutAttention);
        setIdentityProjectionWeights(baseline);
        GradTensor input = GradTensor.of(new float[] {
                0.2f, -0.1f,
                0.4f, 0.3f
        }, 1, 2, 2);

        assertArrayEquals(
                baseline.forward(input).data(),
                dropoutAttention.forward(input).data(),
                1e-6f);
    }

    @Test
    void testMultiHeadAttentionCausalMask() {
        var mha = new MultiHeadAttention(256, 8);
        var input = GradTensor.randn(2, 5, 256);
        var output = mha.forward(input, true);
        assertArrayEquals(new long[]{2, 5, 256}, output.shape());
    }

    @Test
    void testFlashAttentionCreation() {
        var fa = new FlashAttention(512, 8);
        assertNotNull(fa);
    }

    @Test
    void flashAttentionTrainingPathBackpropagates() {
        var fa = new FlashAttention(2, 1);
        setIdentityProjectionWeights(fa);
        float[] values = new float[] {0.2f, -0.1f, 0.4f, 0.3f};
        var input = GradTensor.of(values, 1, 2, 2).requiresGrad(true);

        fa.forward(input).sum().backward();

        assertNotNull(input.grad());
        assertNotNull(fa.namedParameters().get("wQ.weight").grad());
        assertNotNull(fa.namedParameters().get("wK.weight").grad());
        assertNotNull(fa.namedParameters().get("wV.weight").grad());
        assertNotNull(fa.namedParameters().get("wO.weight").grad());
        assertArrayEquals(
                finiteDifferenceInputGradient(fa, values),
                input.grad().data(),
                3e-3f);
    }

    @Test
    void flashAttentionTiledInferenceMatchesStandardAttentionHeadMerge() {
        var flash = new FlashAttention(4, 2, false, 1);
        var standard = new MultiHeadAttention(4, 2);
        setIdentityProjectionWeights(flash, 4);
        setIdentityProjectionWeights(standard, 4);
        float[] values = new float[] {
                0.2f, -0.1f, 0.4f, 0.3f,
                -0.2f, 0.5f, 0.1f, -0.4f
        };

        float[] flashOut;
        float[] standardOut;
        try (NoGrad ignored = NoGrad.enter()) {
            flashOut = flash.forward(GradTensor.of(values, 1, 2, 4)).data();
            standardOut = standard.forward(GradTensor.of(values, 1, 2, 4)).data();
        }

        assertArrayEquals(standardOut, flashOut, 1e-5f);
    }

    @Test
    void testPositionalEncoding() {
        var pe = new PositionalEncoding(128, 512);
        var input = GradTensor.randn(2, 10, 128);
        var output = pe.forward(input);
        assertArrayEquals(new long[]{2, 10, 128}, output.shape());
    }

    @Test
    void testTransformerBlock() {
        var block = new TransformerBlock(256, 4, 512, 0.1f);
        var input = GradTensor.randn(2, 10, 256);
        var output = block.forward(input);
        assertArrayEquals(new long[]{2, 10, 256}, output.shape());
    }

    @Test
    void transformerBlockUsesGeluFeedForwardActivation() {
        var block = new TransformerBlock(2, 1, 2, 0f);
        var params = block.namedParameters();
        float[] zero2x2 = new float[4];
        float[] zeroBias = new float[2];

        params.get("wQ.weight").setData(GradTensor.of(zero2x2, 2, 2));
        params.get("wK.weight").setData(GradTensor.of(zero2x2, 2, 2));
        params.get("wV.weight").setData(GradTensor.of(zero2x2, 2, 2));
        params.get("wO.weight").setData(GradTensor.of(zero2x2, 2, 2));
        params.get("wO.bias").setData(GradTensor.of(zeroBias, 1, 2));
        params.get("norm2.weight").setData(GradTensor.of(zeroBias, 1, 2));
        params.get("norm2.bias").setData(GradTensor.of(new float[] {-1f, 1f}, 1, 2));
        params.get("ff1.weight").setData(GradTensor.of(identity(2), 2, 2));
        params.get("ff1.bias").setData(GradTensor.of(zeroBias, 1, 2));
        params.get("ff2.weight").setData(GradTensor.of(identity(2), 2, 2));
        params.get("ff2.bias").setData(GradTensor.of(zeroBias, 1, 2));

        GradTensor output = block.forward(GradTensor.of(new float[] {0f, 0f}, 1, 1, 2));

        assertInstanceOf(GELU.class, block.modules().get("activation"));
        assertArrayEquals(new float[] {gelu(-1f), gelu(1f)}, output.data(), 1e-5f);
    }

    @Test
    void testTransformerEncoderLayer() {
        var layer = new TransformerEncoderLayer(256, 4, 512, 0.1f);
        var input = GradTensor.randn(2, 10, 256);
        var output = layer.forward(input);
        assertArrayEquals(new long[]{2, 10, 256}, output.shape());
    }

    @Test
    void transformerLayersPassDropoutProbabilityToAttentionModules() {
        var encoder = new TransformerEncoderLayer(4, 2, 8, 0.25f);
        var decoder = new TransformerDecoderLayer(4, 2, 8, 0.35f);

        assertEquals(0.25f, ((Dropout) encoder.modules().get("self_attn.attn_dropout")).getP(), 1e-6f);
        assertEquals(0.35f, ((Dropout) decoder.modules().get("self_attn.attn_dropout")).getP(), 1e-6f);
        assertEquals(0.35f, ((Dropout) decoder.modules().get("cross_attn.attn_dropout")).getP(), 1e-6f);
    }

    @Test
    void testCausalMaskCreation() {
        var mask = MultiHeadAttention.createCausalMask(5, 5);
        assertArrayEquals(new long[]{5, 5}, mask.shape());
        // Lower triangular: position (i,j) should be 1 if j <= i
        assertEquals(1f, mask.item(0), 1e-5f);  // (0,0)
        assertEquals(0f, mask.item(1), 1e-5f);  // (0,1) - masked
    }

    private static void setIdentityProjectionWeights(MultiHeadAttention mha) {
        setIdentityProjectionWeights(mha, 2);
    }

    private static void setIdentityProjectionWeights(MultiHeadAttention mha, int dim) {
        var params = mha.namedParameters();
        float[] identity = identity(dim);
        float[] zeroBias = new float[dim];
        params.get("q_proj.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("k_proj.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("v_proj.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("out_proj.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("q_proj.bias").setData(GradTensor.of(zeroBias, 1, dim));
        params.get("k_proj.bias").setData(GradTensor.of(zeroBias, 1, dim));
        params.get("v_proj.bias").setData(GradTensor.of(zeroBias, 1, dim));
        params.get("out_proj.bias").setData(GradTensor.of(zeroBias, 1, dim));
    }

    private static void setIdentityProjectionWeights(FlashAttention fa) {
        setIdentityProjectionWeights(fa, 2);
    }

    private static void setIdentityProjectionWeights(FlashAttention fa, int dim) {
        var params = fa.namedParameters();
        float[] identity = identity(dim);
        float[] zeroBias = new float[dim];
        params.get("wQ.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("wK.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("wV.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("wO.weight").setData(GradTensor.of(identity, dim, dim));
        params.get("wO.bias").setData(GradTensor.of(zeroBias, 1, dim));
    }

    private static float[] identity(int dim) {
        float[] identity = new float[dim * dim];
        for (int i = 0; i < dim; i++) {
            identity[i * dim + i] = 1f;
        }
        return identity;
    }

    private static float gelu(float x) {
        float sqrt2OverPi = (float) Math.sqrt(2.0 / Math.PI);
        float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + (float) Math.tanh(inner));
    }

    private static float[] finiteDifferenceInputGradient(MultiHeadAttention mha, float[] values, boolean causal) {
        float[] grad = new float[values.length];
        float eps = 1e-3f;
        for (int i = 0; i < values.length; i++) {
            float[] plus = values.clone();
            float[] minus = values.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (forwardSum(mha, plus, causal) - forwardSum(mha, minus, causal)) / (2f * eps);
        }
        return grad;
    }

    private static float forwardSum(MultiHeadAttention mha, float[] values, boolean causal) {
        float sum = 0f;
        float[] out = mha.forward(GradTensor.of(values, 1, 2, 2), causal).data();
        for (float value : out) {
            sum += value;
        }
        return sum;
    }

    private static float[] finiteDifferenceInputGradient(FlashAttention fa, float[] values) {
        float[] grad = new float[values.length];
        float eps = 1e-3f;
        for (int i = 0; i < values.length; i++) {
            float[] plus = values.clone();
            float[] minus = values.clone();
            plus[i] += eps;
            minus[i] -= eps;
            grad[i] = (forwardSum(fa, plus) - forwardSum(fa, minus)) / (2f * eps);
        }
        return grad;
    }

    private static float forwardSum(FlashAttention fa, float[] values) {
        float sum = 0f;
        float[] out = fa.forward(GradTensor.of(values, 1, 2, 2)).data();
        for (float value : out) {
            sum += value;
        }
        return sum;
    }
}
