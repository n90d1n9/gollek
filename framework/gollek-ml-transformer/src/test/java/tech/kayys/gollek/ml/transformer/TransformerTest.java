package tech.kayys.gollek.ml.transformer;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

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
    void testTransformerEncoderLayer() {
        var layer = new TransformerEncoderLayer(256, 4, 512, 0.1f);
        var input = GradTensor.randn(2, 10, 256);
        var output = layer.forward(input);
        assertArrayEquals(new long[]{2, 10, 256}, output.shape());
    }

    @Test
    void testCausalMaskCreation() {
        var mask = MultiHeadAttention.createCausalMask(5, 5);
        assertArrayEquals(new long[]{5, 5}, mask.shape());
        // Lower triangular: position (i,j) should be 1 if j <= i
        assertEquals(1f, mask.item(0), 1e-5f);  // (0,0)
        assertEquals(0f, mask.item(1), 1e-5f);  // (0,1) - masked
    }
}
