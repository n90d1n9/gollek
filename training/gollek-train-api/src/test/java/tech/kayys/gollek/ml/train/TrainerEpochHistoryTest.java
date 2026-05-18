package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerEpochHistoryTest {

    @Test
    void recordsTrainAndValidationIntoSingleFlattenedEpochRow() {
        TrainerEpochHistory history = new TrainerEpochHistory();

        history.recordTrain(new TrainerEpochHistory.TrainRecord(
                0,
                5.0,
                0.1,
                2,
                3,
                10.0,
                8.0,
                4.0,
                3.0,
                2,
                12L,
                true,
                6.0,
                2.5,
                4,
                20L,
                true,
                16.0,
                false,
                0,
                new ThroughputSnapshot(2L, 4L, 8L, 4L, 2_000_000_000L),
                Map.of("mae", 2.0),
                Map.of("detail", Map.of("tp", 3L))));
        history.recordValidation(new TrainerEpochHistory.ValidationRecord(
                0,
                4.0,
                0.05,
                4,
                new ThroughputSnapshot(1L, 2L, 4L, 2L, 1_000_000_000L),
                Map.of("mae", 1.5),
                Map.of("detail", Map.of("tp", 2L)),
                1.5,
                "validationMetric.mae",
                "MIN"));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(1, rows.size());
        Map<String, Object> row = rows.get(0);

        assertEquals(0, row.get("epoch"));
        assertEquals(5.0, row.get("trainLoss"));
        assertEquals(4.0, row.get("validationLoss"));
        assertEquals(0.05, row.get("learningRate"));
        assertEquals(2, row.get("optimizerStepCount"));
        assertEquals(4, row.get("schedulerStepCount"));
        assertEquals(2.0, row.get("trainMetric.mae"));
        assertEquals(1.5, row.get("validationMetric.mae"));
        assertEquals("validationMetric.mae", row.get("bestModelMonitor"));
        assertEquals("MIN", row.get("bestModelMonitorMode"));
        assertEquals(1.5, row.get("bestModelMonitorValue"));
        assertEquals(2L, row.get("trainBatchCount"));
        assertEquals(1L, row.get("validationBatchCount"));
        assertEquals(16.0, row.get("mixedPrecisionLossScale"));
    }

    @Test
    void snapshotIsImmutableAndNestedMapsAreCopied() {
        TrainerEpochHistory history = new TrainerEpochHistory();
        Map<String, Object> detail = new LinkedHashMap<>();
        detail.put("tp", 1L);

        history.recordTrain(new TrainerEpochHistory.TrainRecord(
                1,
                1.0,
                0.1,
                1,
                1,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0L,
                false,
                0.0,
                0.0,
                0,
                0L,
                false,
                Double.NaN,
                false,
                0,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of("loss", 1.0),
                Map.of("detail", detail)));

        List<Map<String, Object>> rows = history.snapshot();
        Map<String, Object> row = rows.get(0);
        @SuppressWarnings("unchecked")
        Map<String, Object> details = (Map<String, Object>) row.get("trainMetricDetails.detail");

        detail.put("fp", 2L);

        assertEquals(Map.of("tp", 1L), details);
        assertThrows(UnsupportedOperationException.class, () -> rows.add(Map.of()));
        assertThrows(UnsupportedOperationException.class, () -> row.put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> details.put("x", 1));
    }

    @Test
    void replaceWithLoadedRowsCopiesInputAndValidationCanMerge() {
        TrainerEpochHistory history = new TrainerEpochHistory();
        Map<String, Object> loaded = new LinkedHashMap<>();
        loaded.put("epoch", 2);
        loaded.put("trainLoss", 3.0);

        history.replaceWith(List.of(loaded));
        loaded.put("epoch", 99);
        history.recordValidation(new TrainerEpochHistory.ValidationRecord(
                2,
                2.0,
                0.01,
                5,
                new ThroughputSnapshot(1L, 1L, 1L, 1L, 1_000_000L),
                Map.of(),
                Map.of(),
                Double.NaN,
                "validation_loss",
                "MIN"));

        List<Map<String, Object>> rows = history.snapshot();
        assertEquals(1, rows.size());
        assertEquals(2, rows.get(0).get("epoch"));
        assertEquals(3.0, rows.get(0).get("trainLoss"));
        assertEquals(2.0, rows.get(0).get("validationLoss"));
    }
}
