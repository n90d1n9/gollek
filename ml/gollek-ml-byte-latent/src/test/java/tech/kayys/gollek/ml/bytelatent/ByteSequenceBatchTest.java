package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteSequenceBatchTest {

    @Test
    void defensivelyCopiesSequences() {
        byte[] source = new byte[] {65, 66, 67};
        ByteSequenceBatch batch = new ByteSequenceBatch(List.of(source));

        source[0] = 90;
        assertEquals(65, batch.sequences().getFirst()[0]);

        byte[] exposed = batch.sequences().getFirst();
        exposed[1] = 88;
        assertEquals(66, batch.sequences().getFirst()[1]);
    }
}
