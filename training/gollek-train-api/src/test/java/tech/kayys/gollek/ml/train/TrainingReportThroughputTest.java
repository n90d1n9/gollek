package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportThroughputTest {
    @Test
    void readsTypedThroughputFromReportMetadata() {
        TrainingReport report = report(Map.ofEntries(
                Map.entry("epochHistory", List.of(Map.of(
                        "epoch", 0,
                        "trainLoss", 0.7,
                        "validationLoss", 0.8,
                        "learningRate", 0.01))),
                Map.entry("trainBatchCount", 4L),
                Map.entry("trainSampleCount", 16L),
                Map.entry("trainInputElementCount", 64L),
                Map.entry("trainLabelElementCount", 16L),
                Map.entry("trainComputeMillis", 2_000.0),
                Map.entry("trainSamplesPerSecond", 8.0),
                Map.entry("trainBatchesPerSecond", 2.0),
                Map.entry("trainAverageBatchMillis", 500.0),
                Map.entry("validationBatchCount", 2L),
                Map.entry("validationSampleCount", 6L),
                Map.entry("validationInputElementCount", 24L),
                Map.entry("validationLabelElementCount", 6L),
                Map.entry("validationComputeMillis", 1_500.0),
                Map.entry("validationSamplesPerSecond", 4.0),
                Map.entry("validationBatchesPerSecond", 4.0 / 3.0),
                Map.entry("validationAverageBatchMillis", 750.0)));

        TrainingReportThroughput throughput = report.throughput();
        Map<String, Object> map = report.throughputMap();

        assertTrue(throughput.available());
        assertEquals(4L, throughput.train().batchCount().orElseThrow());
        assertEquals(16L, throughput.train().sampleCount().orElseThrow());
        assertEquals(64L, throughput.train().inputElementCount().orElseThrow());
        assertEquals(16L, throughput.train().labelElementCount().orElseThrow());
        assertEquals(2_000.0, throughput.train().computeMillis().orElseThrow(), 1e-12);
        assertEquals(8.0, throughput.train().samplesPerSecond().orElseThrow(), 1e-12);
        assertEquals(2.0, throughput.train().batchesPerSecond().orElseThrow(), 1e-12);
        assertEquals(500.0, throughput.train().averageBatchMillis().orElseThrow(), 1e-12);
        assertEquals(750.0, throughput.validation().averageBatchMillis().orElseThrow(), 1e-12);
        assertEquals(Boolean.TRUE, map.get("available"));
        assertEquals(map, Gollek.DL.trainingReportThroughputMap(report));
        assertEquals(throughput, Gollek.DL.trainingReportThroughput(report));
        String markdown = report.throughputMarkdown();
        assertEquals(markdown, Gollek.DL.trainingReportThroughputMarkdown(report));
        assertTrue(markdown.startsWith("## Throughput\n"));
        assertTrue(markdown.contains("| Phase | Batches | Samples | Compute ms | Samples/s | Batches/s | Avg batch ms |"));
        assertTrue(markdown.contains("| `train` | 4 | 16 | 2000.000 | 8.000 | 2.000 | 500.000 |"));
        assertTrue(markdown.contains("| `validation` | 2 | 6 | 1500.000 | 4.000 | 1.333 | 750.000 |"));
        assertTrue(report.actionPlanMarkdown().contains("## Throughput"));
    }

    @Test
    void reportsUnavailableWhenThroughputMetadataIsMissing() {
        TrainingReportThroughput throughput = TrainingReportThroughput.fromMetadata(Map.of());

        assertFalse(throughput.available());
        assertFalse(throughput.train().available());
        assertFalse(throughput.validation().available());
        assertEquals(Boolean.FALSE, throughput.toMap().get("available"));
        assertEquals("", TrainingReportThroughputMarkdown.render(throughput));
    }

    private static TrainingReport report(Map<String, Object> metadata) {
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                metadata);
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-06-10T01:02:03Z")));
    }
}
