package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteSequenceCollatorTest {

    @Test
    void packsVariableLengthSequencesForCausalLanguageModeling() {
        ByteSequenceBatch batch = new ByteSequenceBatch(List.of(
                new byte[] {65, 66, 67, 68},
                new byte[] {(byte) 255, 10},
                new byte[] {42}));

        ByteSequenceWindowBatch packed = ByteSequenceCollator.causalLanguageModeling(batch, 4, 999);

        assertEquals(3, packed.batchSize());
        assertEquals(4, packed.sequenceLength());

        assertArrayEquals(new int[] {65, 66, 67, 999}, packed.inputIds()[0]);
        assertArrayEquals(new int[] {66, 67, 68, 999}, packed.targetIds()[0]);
        assertArrayEquals(new boolean[] {true, true, true, false}, packed.attentionMask()[0]);

        assertArrayEquals(new int[] {255, 999, 999, 999}, packed.inputIds()[1]);
        assertArrayEquals(new int[] {10, 999, 999, 999}, packed.targetIds()[1]);
        assertArrayEquals(new boolean[] {true, false, false, false}, packed.attentionMask()[1]);

        assertArrayEquals(new int[] {999, 999, 999, 999}, packed.inputIds()[2]);
        assertArrayEquals(new int[] {999, 999, 999, 999}, packed.targetIds()[2]);
        assertArrayEquals(new boolean[] {false, false, false, false}, packed.attentionMask()[2]);
    }

    @Test
    void truncatesLongerSequencesToRequestedWindowLength() {
        ByteSequenceBatch batch = new ByteSequenceBatch(List.of(
                new byte[] {1, 2, 3, 4, 5, 6}));

        ByteSequenceWindowBatch packed = ByteSequenceCollator.causalLanguageModeling(batch, 3, 0);

        assertArrayEquals(new int[] {1, 2, 3}, packed.inputIds()[0]);
        assertArrayEquals(new int[] {2, 3, 4}, packed.targetIds()[0]);
        assertArrayEquals(new boolean[] {true, true, true}, packed.attentionMask()[0]);
    }
}
