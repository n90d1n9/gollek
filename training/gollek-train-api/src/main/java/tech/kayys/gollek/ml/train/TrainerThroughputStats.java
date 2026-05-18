package tech.kayys.gollek.ml.train;

import java.util.Map;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Accumulates total and per-epoch trainer throughput counters.
 */
final class TrainerThroughputStats {
    private PhaseCounters trainTotal = new PhaseCounters();
    private PhaseCounters validationTotal = new PhaseCounters();
    private PhaseCounters trainEpoch = new PhaseCounters();
    private PhaseCounters validationEpoch = new PhaseCounters();

    synchronized void resetEpoch() {
        trainEpoch = new PhaseCounters();
        validationEpoch = new PhaseCounters();
    }

    synchronized void record(Batch batch, boolean trainPhase, long elapsedNanos) {
        long safeElapsed = Math.max(0L, elapsedNanos);
        long samples = TrainerBatchGuards.sampleCount(batch);
        long inputElements = batch.inputs().numel();
        long labelElements = batch.labels().numel();
        if (trainPhase) {
            trainTotal.add(samples, inputElements, labelElements, safeElapsed);
            trainEpoch.add(samples, inputElements, labelElements, safeElapsed);
        } else {
            validationTotal.add(samples, inputElements, labelElements, safeElapsed);
            validationEpoch.add(samples, inputElements, labelElements, safeElapsed);
        }
    }

    synchronized ThroughputSnapshot trainTotal() {
        return trainTotal.snapshot();
    }

    synchronized ThroughputSnapshot validationTotal() {
        return validationTotal.snapshot();
    }

    synchronized ThroughputSnapshot trainEpoch() {
        return trainEpoch.snapshot();
    }

    synchronized ThroughputSnapshot validationEpoch() {
        return validationEpoch.snapshot();
    }

    static void putPhaseMetadata(Map<String, Object> metadata, String phase, ThroughputSnapshot snapshot) {
        metadata.put(phase + "BatchCount", snapshot.batchCount());
        metadata.put(phase + "SampleCount", snapshot.sampleCount());
        metadata.put(phase + "InputElementCount", snapshot.inputElementCount());
        metadata.put(phase + "LabelElementCount", snapshot.labelElementCount());
        metadata.put(phase + "ComputeMillis", snapshot.computeMillis());
        metadata.put(phase + "SamplesPerSecond", snapshot.samplesPerSecond());
    }

    private static final class PhaseCounters {
        private long batchCount;
        private long sampleCount;
        private long inputElementCount;
        private long labelElementCount;
        private long computeNanos;

        private void add(long samples, long inputElements, long labelElements, long elapsedNanos) {
            batchCount++;
            sampleCount += samples;
            inputElementCount += inputElements;
            labelElementCount += labelElements;
            computeNanos += elapsedNanos;
        }

        private ThroughputSnapshot snapshot() {
            return new ThroughputSnapshot(
                    batchCount,
                    sampleCount,
                    inputElementCount,
                    labelElementCount,
                    computeNanos);
        }
    }
}

record ThroughputSnapshot(
        long batchCount,
        long sampleCount,
        long inputElementCount,
        long labelElementCount,
        long computeNanos) {

    double computeMillis() {
        return computeNanos / 1_000_000.0;
    }

    double samplesPerSecond() {
        if (sampleCount <= 0L || computeNanos <= 0L) {
            return 0.0;
        }
        return sampleCount * 1_000_000_000.0 / computeNanos;
    }

    boolean hasBatchesAndSamples() {
        return batchCount > 0L && sampleCount > 0L;
    }
}
