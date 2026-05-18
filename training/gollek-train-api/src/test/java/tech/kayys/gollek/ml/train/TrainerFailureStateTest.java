package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerFailureStateTest {

    @Test
    void recordsNonFiniteFailureMetadataAndExceptionMessage() {
        TrainerFailureState state = new TrainerFailureState();

        String message = state.recordNonFinite(null, null, Double.POSITIVE_INFINITY, "loss", true);

        assertEquals("unknown loss must be finite, got Infinity", message);
        assertTrue(state.nonFiniteDetected());
        assertTrue(state.terminalFailureDetected());

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                state::throwIfNonFiniteDetected);
        assertEquals(message, error.getMessage());

        Map<String, Object> metadata = new HashMap<>();
        state.putMetadata(metadata);

        assertEquals(Boolean.TRUE, metadata.get("nonFiniteGuardEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteDetected"));
        assertEquals("unknown", metadata.get("nonFinitePhase"));
        assertEquals("value", metadata.get("nonFiniteKind"));
        assertEquals(Double.POSITIVE_INFINITY, metadata.get("nonFiniteValue"));
        assertEquals(Boolean.TRUE, metadata.get("nonFiniteOptimizerStepSkipped"));
        assertEquals("non-finite-unknown-value", metadata.get("stopReason"));
    }

    @Test
    void keepsFirstNonFiniteFailureStable() {
        TrainerFailureState state = new TrainerFailureState();

        state.recordNonFinite("train", "input", Double.NaN, "input", true);
        state.recordNonFinite("validation", "label", Double.NEGATIVE_INFINITY, "label", false);

        Map<String, Object> metadata = new HashMap<>();
        state.putMetadata(metadata);

        assertEquals("train", metadata.get("nonFinitePhase"));
        assertEquals("input", metadata.get("nonFiniteKind"));
        assertTrue(Double.isNaN((Double) metadata.get("nonFiniteValue")));
        assertEquals("non-finite-train-input", metadata.get("stopReason"));
    }

    @Test
    void recordsInvalidBatchAndResetsCleanly() {
        TrainerFailureState state = new TrainerFailureState();

        String message = state.recordInvalidBatch("train", "sample-count-mismatch", "", true);
        assertEquals("train batch is invalid", message);
        assertTrue(state.terminalFailureDetected());

        Map<String, Object> beforeReset = new HashMap<>();
        state.putMetadata(beforeReset);
        assertEquals(Boolean.TRUE, beforeReset.get("invalidBatchDetected"));
        assertEquals("sample-count-mismatch", beforeReset.get("invalidBatchReason"));
        assertEquals("invalid-batch-train-sample-count-mismatch", beforeReset.get("stopReason"));

        state.reset();

        Map<String, Object> afterReset = new HashMap<>();
        state.putMetadata(afterReset);
        assertFalse(state.terminalFailureDetected());
        assertEquals(Boolean.FALSE, afterReset.get("nonFiniteDetected"));
        assertEquals(Boolean.FALSE, afterReset.get("invalidBatchDetected"));
        assertEquals(Boolean.FALSE, afterReset.get("invalidLossShapeDetected"));
        assertEquals(Boolean.FALSE, afterReset.get("invalidMetricDetected"));
    }

    @Test
    void recordsInvalidLossShapeMetadata() {
        TrainerFailureState state = new TrainerFailureState();

        String message = state.recordInvalidLossShape("validation", "[2, 1]", 2L, false);

        assertEquals("validation loss tensor must contain exactly one value before backward, "
                + "got shape [2, 1] with 2 values", message);

        Map<String, Object> metadata = new HashMap<>();
        state.putMetadata(metadata);

        assertEquals(Boolean.TRUE, metadata.get("lossShapeGuardEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("invalidLossShapeDetected"));
        assertEquals("validation", metadata.get("invalidLossShapePhase"));
        assertEquals("[2, 1]", metadata.get("invalidLossShape"));
        assertEquals(2L, metadata.get("invalidLossShapeElementCount"));
        assertEquals(Boolean.FALSE, metadata.get("invalidLossShapeOptimizerStepSkipped"));
        assertEquals("invalid-loss-shape-validation", metadata.get("stopReason"));
    }

    @Test
    void recordsMetricFailureMetadataAndPreservesCause() {
        TrainerFailureState state = new TrainerFailureState();
        RuntimeException cause = new IllegalStateException("boom");

        String message = state.recordInvalidMetricFailure(
                "train",
                "detail_metric",
                "detail",
                "details.bad",
                cause);

        assertEquals("train metric detail_metric detail details.bad failed to produce a value: boom", message);
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                state::throwIfInvalidMetricDetected);
        assertEquals(message, error.getMessage());
        assertSame(cause, error.getCause());

        Map<String, Object> metadata = new HashMap<>();
        state.putMetadata(metadata);

        assertEquals(Boolean.TRUE, metadata.get("failed"));
        assertEquals(Boolean.TRUE, metadata.get("metricFiniteGuardEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("invalidMetricDetected"));
        assertEquals("train", metadata.get("invalidMetricPhase"));
        assertEquals("detail_metric", metadata.get("invalidMetricName"));
        assertTrue(Double.isNaN((Double) metadata.get("invalidMetricValue")));
        assertEquals("detail", metadata.get("invalidMetricKind"));
        assertEquals("details.bad", metadata.get("invalidMetricDetailPath"));
        assertEquals("IllegalStateException", metadata.get("invalidMetricErrorType"));
        assertEquals(Boolean.FALSE, metadata.get("invalidMetricOptimizerStepSkipped"));
        assertEquals("invalid-metric-train-detail_metric", metadata.get("stopReason"));
    }
}
