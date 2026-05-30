package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Materialized mini-batches for one trainer epoch.
 */
public record DiscreteTokenDatasetEpoch(
        List<DiscreteTokenDatasetBatch> batches,
        int exampleCount,
        int emittedExampleCount,
        int droppedExampleCount,
        int requestedBatchSize,
        boolean shuffled,
        boolean dropLast,
        long seed,
        int inputPadToken,
        int targetPadToken) {

    public DiscreteTokenDatasetEpoch {
        Objects.requireNonNull(batches, "batches must not be null");
        batches = List.copyOf(batches);
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
        if (emittedExampleCount < 0) {
            throw new IllegalArgumentException("emittedExampleCount must be >= 0 but was " + emittedExampleCount);
        }
        if (droppedExampleCount < 0) {
            throw new IllegalArgumentException("droppedExampleCount must be >= 0 but was " + droppedExampleCount);
        }
        if (requestedBatchSize <= 0) {
            throw new IllegalArgumentException("requestedBatchSize must be > 0 but was " + requestedBatchSize);
        }
        if (emittedExampleCount + droppedExampleCount != exampleCount) {
            throw new IllegalArgumentException(
                    "emittedExampleCount + droppedExampleCount must equal exampleCount");
        }

        int observedExamples = 0;
        for (int index = 0; index < batches.size(); index++) {
            DiscreteTokenDatasetBatch batch =
                    Objects.requireNonNull(batches.get(index), "batches[" + index + "] must not be null");
            if (batch.inputPadToken() != inputPadToken) {
                throw new IllegalArgumentException("batches[" + index + "] input pad token does not match epoch");
            }
            if (batch.targetPadToken() != targetPadToken) {
                throw new IllegalArgumentException("batches[" + index + "] target pad token does not match epoch");
            }
            int batchSize = batch.batchSize();
            if (batchSize > requestedBatchSize) {
                throw new IllegalArgumentException(
                        "batches[" + index + "] size must be <= requestedBatchSize");
            }
            if (dropLast && batchSize != requestedBatchSize) {
                throw new IllegalArgumentException(
                        "dropLast epochs must contain only full batches; batch " + index + " had size " + batchSize);
            }
            if (!dropLast && index < batches.size() - 1 && batchSize != requestedBatchSize) {
                throw new IllegalArgumentException(
                        "only the last batch may be partial; batch " + index + " had size " + batchSize);
            }
            observedExamples += batchSize;
        }
        if (observedExamples != emittedExampleCount) {
            throw new IllegalArgumentException(
                    "sum of batch sizes must equal emittedExampleCount");
        }
    }

    public int batchCount() {
        return batches.size();
    }

    public boolean hasBatches() {
        return !batches.isEmpty();
    }

    public boolean emittedAllExamples() {
        return droppedExampleCount == 0;
    }

    public long inputTokenCount() {
        long count = 0L;
        for (DiscreteTokenDatasetBatch batch : batches) {
            for (int length : batch.inputLengths()) {
                count += length;
            }
        }
        return count;
    }

    public long targetTokenCount() {
        long count = 0L;
        for (DiscreteTokenDatasetBatch batch : batches) {
            for (int length : batch.targetLengths()) {
                count += length;
            }
        }
        return count;
    }

    public long observedTokenCount() {
        return inputTokenCount() + targetTokenCount();
    }

    public long paddedInputTokenCapacity() {
        long capacity = 0L;
        for (DiscreteTokenDatasetBatch batch : batches) {
            capacity += (long) batch.batchSize() * batch.maxInputLength();
        }
        return capacity;
    }

    public long paddedTargetTokenCapacity() {
        long capacity = 0L;
        for (DiscreteTokenDatasetBatch batch : batches) {
            capacity += (long) batch.batchSize() * batch.maxTargetLength();
        }
        return capacity;
    }

    public long paddedTokenCapacity() {
        return paddedInputTokenCapacity() + paddedTargetTokenCapacity();
    }

    public long inputPaddingTokenCount() {
        return paddedInputTokenCapacity() - inputTokenCount();
    }

    public long targetPaddingTokenCount() {
        return paddedTargetTokenCapacity() - targetTokenCount();
    }

    public long paddingTokenCount() {
        return inputPaddingTokenCount() + targetPaddingTokenCount();
    }

    public double paddingRate() {
        long capacity = paddedTokenCapacity();
        return capacity == 0L ? 0.0d : (double) paddingTokenCount() / capacity;
    }
}
