package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class TrainerBatchFailureRecorderTest {
    private static final Runnable NO_OP = () -> {
    };

    @Test
    void recordsInvalidBatchAndDiscardsGradients() {
        TrainerFailureState failureState = new TrainerFailureState();
        AtomicBoolean discarded = new AtomicBoolean(false);
        TrainerBatchFailureRecorder recorder =
                new TrainerBatchFailureRecorder(failureState, () -> discarded.set(true));

        String message = recorder.invalidBatch("train", "missing-input", "", true);
        recorder.discardPendingGradients();

        Map<String, Object> metadata = metadata(failureState);
        assertEquals("train batch is invalid", message);
        assertEquals(Boolean.TRUE, metadata.get("invalidBatchDetected"));
        assertEquals("missing-input", metadata.get("invalidBatchReason"));
        assertEquals(Boolean.TRUE, metadata.get("invalidBatchOptimizerStepSkipped"));
        assertTrue(discarded.get());
    }

    @Test
    void recordsInvalidLossShape() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerBatchFailureRecorder recorder = new TrainerBatchFailureRecorder(failureState, NO_OP);

        String message = recorder.invalidLossShape("validation", "[2, 1]", 2L, false);

        Map<String, Object> metadata = metadata(failureState);
        assertEquals(
                "validation loss tensor must contain exactly one value before backward, "
                        + "got shape [2, 1] with 2 values",
                message);
        assertEquals(Boolean.TRUE, metadata.get("invalidLossShapeDetected"));
        assertEquals("[2, 1]", metadata.get("invalidLossShape"));
        assertEquals(2L, metadata.get("invalidLossShapeElementCount"));
    }

    @Test
    void recordsNonFiniteTensorFailure() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerBatchFailureRecorder recorder = new TrainerBatchFailureRecorder(failureState, NO_OP);

        String message = recorder.nonFiniteTensor(
                "train",
                "gradient",
                Double.NaN,
                "gradient",
                true,
                5L,
                1L,
                2L,
                0L);

        Map<String, Object> metadata = metadata(failureState);
        assertEquals("train gradient must be finite, got NaN", message);
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteDetected"));
        assertEquals("gradient", metadata.get("nonFiniteKind"));
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteOptimizerStepSkipped"));
        assertEquals(5L, metadata.get("nonFiniteTotalValueCount"));
        assertEquals(3L, metadata.get("nonFiniteValueCount"));
        assertEquals(1L, metadata.get("nonFiniteNanCount"));
        assertEquals(2L, metadata.get("nonFinitePositiveInfinityCount"));
        assertEquals(0L, metadata.get("nonFiniteNegativeInfinityCount"));
    }

    private static Map<String, Object> metadata(TrainerFailureState failureState) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        failureState.putMetadata(metadata);
        return metadata;
    }
}
