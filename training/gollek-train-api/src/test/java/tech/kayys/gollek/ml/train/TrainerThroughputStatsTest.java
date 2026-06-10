package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader.Batch;

class TrainerThroughputStatsTest {

    @Test
    void recordsTotalAndEpochPhaseCounters() {
        TrainerThroughputStats stats = new TrainerThroughputStats();

        stats.record(batch(2, 3), true, 2_000_000_000L);
        stats.record(batch(1, 3), true, 1_000_000_000L);
        stats.record(batch(2, 2), false, 500_000_000L);

        ThroughputSnapshot train = stats.trainTotal();
        assertEquals(2L, train.batchCount());
        assertEquals(3L, train.sampleCount());
        assertEquals(9L, train.inputElementCount());
        assertEquals(3L, train.labelElementCount());
        assertEquals(3_000_000_000L, train.computeNanos());
        assertEquals(3_000.0, train.computeMillis(), 1e-6);
        assertEquals(1.0, train.samplesPerSecond(), 1e-6);
        assertEquals(2.0 / 3.0, train.batchesPerSecond(), 1e-6);
        assertEquals(1_500.0, train.averageBatchMillis(), 1e-6);

        ThroughputSnapshot validation = stats.validationTotal();
        assertEquals(1L, validation.batchCount());
        assertEquals(2L, validation.sampleCount());
        assertEquals(4L, validation.inputElementCount());
        assertEquals(2L, validation.labelElementCount());
        assertEquals(4.0, validation.samplesPerSecond(), 1e-6);
    }

    @Test
    void resetEpochClearsOnlyEpochCounters() {
        TrainerThroughputStats stats = new TrainerThroughputStats();
        stats.record(batch(2, 2), true, 1_000_000L);
        stats.resetEpoch();
        stats.record(batch(1, 2), false, 1_000_000L);

        assertEquals(1L, stats.trainTotal().batchCount());
        assertEquals(0L, stats.trainEpoch().batchCount());
        assertEquals(1L, stats.validationTotal().batchCount());
        assertEquals(1L, stats.validationEpoch().batchCount());
    }

    @Test
    void negativeElapsedTimeIsClampedForRates() {
        TrainerThroughputStats stats = new TrainerThroughputStats();
        stats.record(batch(2, 1), true, -1L);

        ThroughputSnapshot snapshot = stats.trainTotal();
        assertEquals(0L, snapshot.computeNanos());
        assertEquals(0.0, snapshot.samplesPerSecond(), 1e-6);
        assertEquals(true, snapshot.hasBatchesAndSamples());
    }

    @Test
    void writesStablePhaseMetadataKeys() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        TrainerThroughputStats.putPhaseMetadata(
                metadata,
                "train",
                new ThroughputSnapshot(2L, 4L, 8L, 4L, 2_000_000_000L));

        assertEquals(2L, metadata.get("trainBatchCount"));
        assertEquals(4L, metadata.get("trainSampleCount"));
        assertEquals(8L, metadata.get("trainInputElementCount"));
        assertEquals(4L, metadata.get("trainLabelElementCount"));
        assertEquals(2_000.0, (Double) metadata.get("trainComputeMillis"), 1e-6);
        assertEquals(2.0, (Double) metadata.get("trainSamplesPerSecond"), 1e-6);
        assertEquals(1.0, (Double) metadata.get("trainBatchesPerSecond"), 1e-6);
        assertEquals(1_000.0, (Double) metadata.get("trainAverageBatchMillis"), 1e-6);
    }

    private static Batch batch(int samples, int width) {
        int values = samples * width;
        return new Batch(
                GradTensor.of(new float[values], samples, width),
                GradTensor.of(new float[samples], samples));
    }
}
