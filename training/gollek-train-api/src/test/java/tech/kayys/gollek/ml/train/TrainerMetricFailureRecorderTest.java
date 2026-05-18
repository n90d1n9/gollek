package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerMetricFailureRecorderTest {

    @Test
    void recordsInvalidMetricValue() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerMetricFailureRecorder recorder = new TrainerMetricFailureRecorder(failureState);

        String message = recorder.invalidValue("train", "mae", Double.NaN, true);

        Map<String, Object> metadata = metadata(failureState);
        assertEquals("train metric mae must be finite, got NaN", message);
        assertEquals(Boolean.TRUE, metadata.get("invalidMetricDetected"));
        assertEquals("value", metadata.get("invalidMetricKind"));
        assertEquals("mae", metadata.get("invalidMetricName"));
        assertEquals(Boolean.TRUE, metadata.get("invalidMetricOptimizerStepSkipped"));
    }

    @Test
    void recordsInvalidMetricDetail() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerMetricFailureRecorder recorder = new TrainerMetricFailureRecorder(failureState);

        String message = recorder.invalidDetail("validation", "confusion", "details.tp", Double.POSITIVE_INFINITY);

        Map<String, Object> metadata = metadata(failureState);
        assertEquals(
                "validation metric confusion detail details.tp must be finite, got Infinity",
                message);
        assertEquals("detail", metadata.get("invalidMetricKind"));
        assertEquals("details.tp", metadata.get("invalidMetricDetailPath"));
    }

    @Test
    void recordsInvalidMetricFailure() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerMetricFailureRecorder recorder = new TrainerMetricFailureRecorder(failureState);
        IllegalStateException error = new IllegalStateException("broken metric");

        String message = recorder.invalidFailure("validation", "auc", "value", null, error);

        Map<String, Object> metadata = metadata(failureState);
        assertEquals("validation metric auc failed to produce a value: broken metric", message);
        assertEquals("value", metadata.get("invalidMetricKind"));
        assertEquals("IllegalStateException", metadata.get("invalidMetricErrorType"));
    }

    @Test
    void recordsDuplicateMetricName() {
        TrainerFailureState failureState = new TrainerFailureState();
        TrainerMetricFailureRecorder recorder = new TrainerMetricFailureRecorder(failureState);

        String message = recorder.invalidName("train", "accuracy");

        Map<String, Object> metadata = metadata(failureState);
        assertEquals("train metric name must be unique, duplicate: accuracy", message);
        assertEquals("name", metadata.get("invalidMetricKind"));
        assertEquals("accuracy", metadata.get("invalidMetricName"));
    }

    private static Map<String, Object> metadata(TrainerFailureState failureState) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        failureState.putMetadata(metadata);
        return metadata;
    }
}
