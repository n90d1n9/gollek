package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class ByteSequenceWindowBatchTest {

    @Test
    void defensivelyCopiesPackedArrays() {
        int[][] inputs = new int[][] {{1, 2}};
        int[][] targets = new int[][] {{2, 3}};
        boolean[][] mask = new boolean[][] {{true, false}};

        ByteSequenceWindowBatch batch = new ByteSequenceWindowBatch(inputs, targets, mask, 0, 2);

        inputs[0][0] = 99;
        targets[0][0] = 99;
        mask[0][0] = false;

        assertEquals(1, batch.inputIds()[0][0]);
        assertEquals(2, batch.targetIds()[0][0]);
        assertEquals(true, batch.attentionMask()[0][0]);
    }
}
