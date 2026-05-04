package tech.kayys.gollek.ml.autograd;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for GradTensor autograd engine.
 */
class GradTensorAutogradTest {

    @Test
    void testBasicCreation() {
        GradTensor t = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        assertArrayEquals(new long[]{2, 2}, t.shape());
        assertEquals(4, t.numel());
    }

    @Test
    void testZeros() {
        GradTensor t = GradTensor.zeros(3, 4);
        assertEquals(12, t.numel());
        for (float v : t.data()) assertEquals(0f, v);
    }

    @Test
    void testAdd() {
        GradTensor a = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        GradTensor b = GradTensor.of(new float[]{5, 6, 7, 8}, 2, 2);
        GradTensor c = a.add(b);
        assertArrayEquals(new float[]{6, 8, 10, 12}, c.data(), 1e-5f);
    }

    @Test
    void testMatmul() {
        GradTensor a = GradTensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        GradTensor b = GradTensor.of(new float[]{5, 6, 7, 8}, 2, 2);
        GradTensor c = a.matmul(b);
        // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50]
        assertArrayEquals(new float[]{19, 22, 43, 50}, c.data(), 1e-5f);
    }

    @Test
    void testRelu() {
        GradTensor t = GradTensor.of(new float[]{-2, -1, 0, 1, 2}, 5);
        GradTensor r = t.relu();
        assertArrayEquals(new float[]{0, 0, 0, 1, 2}, r.data(), 1e-5f);
    }

    @Test
    void testAutogradScalarBackward() {
        GradTensor x = GradTensor.scalar(3).requiresGrad(true);
        GradTensor y = x.mul(x).sum(); // y = x^2 = 9
        y.backward();
        // dy/dx = 2x = 6
        GradTensor g = x.grad();
        assertNotNull(g);
        assertEquals(6f, g.item(), 1e-3f);
    }

    @Test
    void testAutogradMatmulBackward() {
        GradTensor w = GradTensor.randn(2, 3).requiresGrad(true);
        GradTensor x = GradTensor.randn(4, 2);
        GradTensor y = x.matmul(w);
        GradTensor loss = y.sum();
        loss.backward();
        assertNotNull(w.grad());
        assertEquals(6, w.grad().numel()); // 2x3 = 6
    }

    @Test
    void testChainRule() {
        GradTensor x = GradTensor.scalar(2).requiresGrad(true);
        GradTensor y = x.pow(2);   // y = 4
        GradTensor z = y.mul(3);   // z = 12
        z.backward();
        // dz/dx = 3 * 2x = 12
        assertEquals(12f, x.grad().item(), 1e-3f);
    }

    @Test
    void testNoGrad() {
        assertFalse(NoGrad.isActive());
        try (var ng = NoGrad.enter()) {
            assertTrue(NoGrad.isActive());
        }
        assertFalse(NoGrad.isActive());
    }

    @Test
    void testSoftmax() {
        GradTensor t = GradTensor.of(new float[]{1, 2, 3}, 3);
        GradTensor s = t.softmax();
        float sum = s.data()[0] + s.data()[1] + s.data()[2];
        assertEquals(1f, sum, 1e-5f);
        assertTrue(s.data()[2] > s.data()[1] && s.data()[1] > s.data()[0]);
    }

    @Test
    void testCat() {
        GradTensor a = GradTensor.ones(2, 3);
        GradTensor b = GradTensor.zeros(2, 3);
        GradTensor c = GradTensor.cat(a, b);
        assertArrayEquals(new long[]{4, 3}, c.shape());
        assertEquals(1f, c.item(0));
        assertEquals(0f, c.item(6));
    }

    @Test
    void testStack() {
        GradTensor a = GradTensor.ones(2, 3);
        GradTensor b = GradTensor.zeros(2, 3);
        GradTensor c = GradTensor.stack(0, a, b);
        assertArrayEquals(new long[]{2, 2, 3}, c.shape());
    }

    @Test
    void testReshape() {
        GradTensor t = GradTensor.arange(0, 12, 1);
        GradTensor r = t.reshape(3, 4);
        assertArrayEquals(new long[]{3, 4}, r.shape());
    }

    @Test
    void testDetach() {
        GradTensor x = GradTensor.scalar(5).requiresGrad(true);
        GradTensor y = x.detach();
        assertFalse(y.requiresGrad());
        assertNull(y.gradFn());
    }

    @Test
    void testClamp() {
        GradTensor t = GradTensor.of(new float[]{-2, -1, 0, 1, 2, 3}, 6);
        GradTensor c = t.clamp(0, 2);
        assertArrayEquals(new float[]{0, 0, 0, 1, 2, 2}, c.data(), 1e-5f);
    }

    @Test
    void testSigmoid() {
        GradTensor t = GradTensor.scalar(0);
        GradTensor s = t.sigmoid();
        assertEquals(0.5f, s.item(), 1e-5f);
    }

    @Test
    void testBackwardOnNonScalarFails() {
        GradTensor t = GradTensor.randn(3, 4);
        assertThrows(IllegalStateException.class, t::backward);
    }
}
