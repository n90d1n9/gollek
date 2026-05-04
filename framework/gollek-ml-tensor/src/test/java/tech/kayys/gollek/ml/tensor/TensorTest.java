package tech.kayys.gollek.ml.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void testZeros() {
        Tensor t = Tensor.zeros(2, 3);
        assertEquals(6, t.numel());
        assertArrayEquals(new long[]{2, 3}, t.shape());
        for (float v : t.data()) assertEquals(0f, v);
    }

    @Test
    void testOnes() {
        Tensor t = Tensor.ones(3, 2);
        assertEquals(6, t.numel());
        for (float v : t.data()) assertEquals(1f, v);
    }

    @Test
    void testRandn() {
        Tensor t = Tensor.randn(100);
        assertEquals(100, t.numel());
        assertEquals(1, t.ndim());
    }

    @Test
    void testEye() {
        Tensor t = Tensor.eye(3);
        assertArrayEquals(new long[]{3, 3}, t.shape());
        assertEquals(1f, t.item(0));
        assertEquals(0f, t.item(1));
        assertEquals(1f, t.item(4));
    }

    @Test
    void testReshape() {
        Tensor t = Tensor.arange(12);
        Tensor r = t.reshape(3, 4);
        assertArrayEquals(new long[]{3, 4}, r.shape());
        assertEquals(12, r.numel());
    }

    @Test
    void testTranspose() {
        Tensor t = Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        Tensor tr = t.transpose();
        assertEquals(1f, tr.item(0));
        assertEquals(3f, tr.item(1));
        assertEquals(2f, tr.item(2));
        assertEquals(4f, tr.item(3));
    }

    @Test
    void testDevice() {
        Tensor t = Tensor.randn(10);
        assertTrue(t.isCpu());
        Tensor onCpu = t.to(tech.kayys.gollek.runtime.tensor.Device.CPU);
        assertEquals(tech.kayys.gollek.runtime.tensor.Device.CPU, onCpu.device());
    }

    @Test
    void testToString() {
        Tensor t = Tensor.randn(3, 4);
        assertNotNull(t.toString());
        assertTrue(t.toString().contains("shape"));
        assertTrue(t.toString().contains("device"));
    }
}
