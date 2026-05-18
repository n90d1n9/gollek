package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class TextByteSequenceDatasetTest {

    @Test
    void batchesUtf8TextIntoByteSequenceBatch() {
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of("halo", "dunia"));

        assertEquals(2, dataset.size());
        assertArrayEquals("halo".getBytes(StandardCharsets.UTF_8), dataset.get(0));

        ByteSequenceBatch batch = dataset.batch(0, 2);
        assertEquals(2, batch.batchSize());
        assertArrayEquals("dunia".getBytes(StandardCharsets.UTF_8), batch.sequences().get(1));
    }

    @Test
    void loadsDatasetFromLineFile() throws IOException {
        Path file = Files.createTempFile("gollek-byte-latent", ".txt");
        Files.write(file, List.of("alpha", "beta"), StandardCharsets.UTF_8);

        TextByteSequenceDataset dataset = TextByteSequenceDataset.fromLines(file);

        assertEquals(2, dataset.size());
        assertArrayEquals("alpha".getBytes(StandardCharsets.UTF_8), dataset.get(0));
        assertEquals(StandardCharsets.UTF_8, dataset.charset());
    }
}
