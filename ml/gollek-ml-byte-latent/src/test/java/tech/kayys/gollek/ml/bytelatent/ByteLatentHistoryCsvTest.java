package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class ByteLatentHistoryCsvTest {

    @Test
    void writesAndReadsEpochHistoryRows() throws Exception {
        Path file = Files.createTempFile("gollek-byte-history", ".csv");
        List<ByteLatentHistoryRow> rows = List.of(
                new ByteLatentHistoryRow(1, 2, 2, 0.5d),
                new ByteLatentHistoryRow(2, 4, 2, 0.25d));

        ByteLatentHistoryCsv.write(file, rows);

        assertEquals(rows, ByteLatentHistoryCsv.read(file));
    }
}
