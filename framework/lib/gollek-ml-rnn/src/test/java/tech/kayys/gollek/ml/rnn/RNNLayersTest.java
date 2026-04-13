package tech.kayys.gollek.ml.rnn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class RNNLayersTest {

    @Test
    void testLSTMCellCreation() {
        var cell = new tech.kayys.gollek.ml.rnn.cells.LSTMCell(64, 32);
        assertNotNull(cell);
    }

    @Test
    void testGRUCellCreation() {
        var cell = new tech.kayys.gollek.ml.rnn.cells.GRUCell(64, 32);
        assertNotNull(cell);
    }

    @Test
    void testLSTMLayer() {
        var layer = new tech.kayys.gollek.ml.rnn.LSTM(64, 32);
        var input = GradTensor.randn(2, 5, 64); // [batch, seq, input_dim]
        var output = layer.forward(input);
        assertTrue(output.ndim() >= 2);
    }

    @Test
    void testGRULayer() {
        var layer = new tech.kayys.gollek.ml.rnn.GRU(64, 32);
        var input = GradTensor.randn(2, 5, 64);
        var output = layer.forward(input);
        assertTrue(output.ndim() >= 2);
    }

    @Test
    void testBidirectional() {
        var bi = new tech.kayys.gollek.ml.rnn.Bidirectional(new tech.kayys.gollek.ml.rnn.LSTM(64, 32));
        assertNotNull(bi);
    }

    @Test
    void testRNNCellForward() {
        var cell = new tech.kayys.gollek.ml.rnn.cells.GRUCell(64, 32);
        var input = GradTensor.randn(2, 64);
        var hPrev = GradTensor.zeros(2, 32);
        var output = cell.forward(input, hPrev);
        assertArrayEquals(new long[]{2, 32}, output.shape());
    }
}
