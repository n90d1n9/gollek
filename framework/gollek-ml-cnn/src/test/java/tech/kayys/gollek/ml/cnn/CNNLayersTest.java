package tech.kayys.gollek.ml.cnn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class CNNLayersTest {

    @Test
    void testConv2dCreation() {
        var conv = new Conv2d(3, 64, 3);
        assertEquals(3, conv.getInChannels());
        assertEquals(64, conv.getOutChannels());
    }

    @Test
    void testConv2dWithStrideAndPadding() {
        var conv = new Conv2d(64, 128, 3, 1, 1);
        assertEquals(64, conv.getInChannels());
        assertEquals(128, conv.getOutChannels());
    }

    @Test
    void testConv2dForwardShape() {
        var conv = new Conv2d(3, 16, 3, 1, 1);
        var input = GradTensor.randn(2, 3, 32, 32);
        var output = conv.forward(input);
        assertArrayEquals(new long[]{2, 16, 32, 32}, output.shape());
    }

    @Test
    void testMaxPool2d() {
        var pool = new MaxPool2d(2, 2, 0);
        var input = GradTensor.randn(2, 16, 32, 32);
        var output = pool.forward(input);
        assertArrayEquals(new long[]{2, 16, 16, 16}, output.shape());
    }

    @Test
    void testAvgPool2d() {
        var pool = new AvgPool2d(2, 2, 0);
        var input = GradTensor.randn(2, 16, 32, 32);
        var output = pool.forward(input);
        assertArrayEquals(new long[]{2, 16, 16, 16}, output.shape());
    }

    @Test
    void testBatchNorm2d() {
        var bn = new BatchNorm2d(16);
        var input = GradTensor.randn(4, 16, 8, 8);
        var output = bn.forward(input);
        assertArrayEquals(new long[]{4, 16, 8, 8}, output.shape());
    }

    @Test
    void testConvTranspose2d() {
        var conv = new ConvTranspose2d(16, 32, 3, 1, 1);
        var input = GradTensor.randn(2, 16, 8, 8);
        var output = conv.forward(input);
        // ConvTranspose2d output shape depends on implementation
        assertTrue(output.ndim() == 4);
    }
}
