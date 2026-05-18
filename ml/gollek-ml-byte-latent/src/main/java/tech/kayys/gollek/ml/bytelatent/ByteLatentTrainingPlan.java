package tech.kayys.gollek.ml.bytelatent;

import java.util.List;
import java.util.Objects;

/**
 * Materialized trainer-ready byte-latent batches for a dataset/config pair.
 */
public record ByteLatentTrainingPlan(
        ByteLatentTrainerConfig config,
        int exampleCount,
        int batchCount,
        List<ByteSequenceWindowBatch> batches) {

    public ByteLatentTrainingPlan {
        config = Objects.requireNonNull(config, "config must not be null");
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
        if (batchCount < 0) {
            throw new IllegalArgumentException("batchCount must be >= 0 but was " + batchCount);
        }
        batches = List.copyOf(Objects.requireNonNull(batches, "batches must not be null"));
    }
}
