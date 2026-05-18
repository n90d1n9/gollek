package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerLearningRateSchedulerMetadataTest {

    @TempDir
    Path tempDir;

    @Test
    void publishesEnabledSchedulerStateAndCheckpoint() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("scheduler.json"), "{}");
        Map<String, Object> metadata = new HashMap<>();

        TrainerLearningRateSchedulerMetadata.put(
                metadata,
                true,
                "VALIDATION",
                7,
                "ReduceLROnPlateau",
                "validationMetric.f1",
                true,
                Map.of("stepCount", 7, "currentLr", 0.025),
                0.025,
                new TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint(
                        true,
                        true,
                        true,
                        checkpoint,
                        false,
                        true,
                        true,
                        null,
                        null));

        assertEquals(Boolean.TRUE, metadata.get("learningRateSchedulerEnabled"));
        assertEquals("VALIDATION", metadata.get("learningRateSchedulerStepUnit"));
        assertEquals(7, metadata.get("learningRateSchedulerStepCount"));
        assertEquals("ReduceLROnPlateau", metadata.get("learningRateSchedulerType"));
        assertEquals("validationMetric.f1", metadata.get("learningRateSchedulerMonitor"));
        assertEquals(Boolean.TRUE, metadata.get("learningRateSchedulerMonitorMetricDriven"));
        assertEquals(Map.of("stepCount", 7, "currentLr", 0.025), metadata.get("learningRateSchedulerState"));
        assertEquals(7, metadata.get("learningRateSchedulerState.stepCount"));
        assertEquals(0.025, metadata.get("learningRateSchedulerState.currentLr"));
        assertEquals(0.025, metadata.get("learningRate"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointSupported"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointResumeRequested"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointPresent"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointMissingOnResume"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointSaved"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointLoadFailed"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointSaveFailed"));
    }

    @Test
    void publishesDisabledSchedulerFailureState() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerLearningRateSchedulerMetadata.put(
                metadata,
                false,
                "BATCH",
                0,
                "none",
                "validation_loss",
                false,
                null,
                0.1,
                new TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint(
                        false,
                        false,
                        false,
                        null,
                        true,
                        false,
                        false,
                        "load failed",
                        "save failed"));

        assertEquals(Boolean.FALSE, metadata.get("learningRateSchedulerEnabled"));
        assertEquals("BATCH", metadata.get("learningRateSchedulerStepUnit"));
        assertEquals(0, metadata.get("learningRateSchedulerStepCount"));
        assertEquals("none", metadata.get("learningRateSchedulerType"));
        assertEquals("validation_loss", metadata.get("learningRateSchedulerMonitor"));
        assertEquals(Boolean.FALSE, metadata.get("learningRateSchedulerMonitorMetricDriven"));
        assertEquals(Map.of(), metadata.get("learningRateSchedulerState"));
        assertEquals(0.1, metadata.get("learningRate"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointSupported"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointResumeRequested"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointPresent"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointMissingOnResume"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointLoaded"));
        assertEquals(Boolean.FALSE, metadata.get("schedulerCheckpointSaved"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointLoadFailed"));
        assertEquals(Boolean.TRUE, metadata.get("schedulerCheckpointSaveFailed"));
    }
}
