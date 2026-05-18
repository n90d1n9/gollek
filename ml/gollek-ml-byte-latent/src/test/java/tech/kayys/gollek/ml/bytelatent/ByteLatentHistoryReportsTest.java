package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ByteLatentHistoryReportsTest {

    @Test
    void supportsCompactAggregateSelectors() {
        List<ByteLatentHistoryRow> rows = sampleRows();

        assertEquals(0.5d, ByteLatentHistoryReports.select(rows, "history:meanLoss"));
        assertEquals(0.5d, ByteLatentHistoryReports.select(rows, "history:avgLoss"));
        assertEquals(0.25d, ByteLatentHistoryReports.select(rows, "history:minLoss"));
        assertEquals(0.75d, ByteLatentHistoryReports.select(rows, "history:maxLoss"));
        assertEquals(0.25d, ByteLatentHistoryReports.select(rows, "history:lastLoss"));
        assertEquals(3, ByteLatentHistoryReports.select(rows, "history:lastEpoch"));
        assertEquals(6, ByteLatentHistoryReports.select(rows, "history:lastGlobalStep"));
        assertEquals(6, ByteLatentHistoryReports.select(rows, "history:totalBatches"));
    }

    @Test
    void supportsSummaryAndSelectorChains() {
        List<ByteLatentHistoryRow> rows = sampleRows();

        Object value = ByteLatentHistoryReports.select(rows, "history:summary");
        assertInstanceOf(Map.class, value);
        @SuppressWarnings("unchecked")
        Map<String, Object> summary = (Map<String, Object>) value;
        assertEquals(3, summary.get("count"));
        assertEquals(0.5d, summary.get("meanLoss"));
        assertEquals(0.25d, summary.get("lastLoss"));
        assertEquals(3, summary.get("lastEpoch"));

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> sorted = (List<Map<String, Object>>) ByteLatentHistoryReports.select(
                rows, "history:sort=-trainLoss:top=2");
        assertEquals(2, sorted.size());
        assertEquals(2, sorted.get(0).get("epoch"));
        assertEquals(0.75d, sorted.get(0).get("trainLoss"));

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> filtered = (List<Map<String, Object>>) ByteLatentHistoryReports.select(
                rows, "history:epoch=2");
        assertEquals(1, filtered.size());
        assertEquals(2, filtered.get(0).get("epoch"));
    }

    @Test
    void loadsAndSelectsFromPersistedHistoryFile() throws Exception {
        Path file = Files.createTempFile("gollek-byte-history-report", ".csv");
        ByteLatentHistoryCsv.write(file, sampleRows());

        assertEquals(3, ByteLatentHistoryReports.select(file, "historyCount"));
        assertEquals(0.25d, ByteLatentHistoryReports.select(file, "history:lastLoss"));
    }

    private static List<ByteLatentHistoryRow> sampleRows() {
        return List.of(
                new ByteLatentHistoryRow(1, 2, 2, 0.5d),
                new ByteLatentHistoryRow(2, 4, 2, 0.75d),
                new ByteLatentHistoryRow(3, 6, 2, 0.25d));
    }
}
