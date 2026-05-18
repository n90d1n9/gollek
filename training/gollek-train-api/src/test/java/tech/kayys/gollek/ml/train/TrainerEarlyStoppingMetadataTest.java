package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerEarlyStoppingMetadataTest {

    @Test
    void publishesConfigurationAndMonitorState() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerEarlyStoppingMetadata.put(
                metadata,
                3,
                0.25,
                "validationMetric.f1",
                CanonicalTrainer.BestModelMonitorMode.MAX,
                true,
                new TrainerEarlyStoppingMetadata.MonitorState(2, 0.8, 0.7, 1),
                false,
                -1);

        assertEquals(Boolean.TRUE, metadata.get("earlyStoppingEnabled"));
        assertEquals(3, metadata.get("earlyStoppingPatience"));
        assertEquals(0.25, metadata.get("earlyStoppingMinDelta"));
        assertEquals("validationMetric.f1", metadata.get("earlyStoppingMonitor"));
        assertEquals("MAX", metadata.get("earlyStoppingMonitorMode"));
        assertEquals(Boolean.TRUE, metadata.get("earlyStoppingMonitorMetricDriven"));
        assertEquals(2, metadata.get("earlyStoppingMonitorBestEpoch"));
        assertEquals(0.8, metadata.get("earlyStoppingMonitorBestValue"));
        assertEquals(0.7, metadata.get("earlyStoppingMonitorLatestValue"));
        assertEquals(1, metadata.get("earlyStoppingMonitorEpochsWithoutImprovement"));
        assertFalse(metadata.containsKey("earlyStoppingTriggered"));
        assertFalse(metadata.containsKey("stopReason"));
    }

    @Test
    void publishesTriggeredStopReasonAndClampsDisabledConfiguration() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerEarlyStoppingMetadata.put(
                metadata,
                -2,
                -0.5,
                "validation_loss",
                CanonicalTrainer.BestModelMonitorMode.MIN,
                false,
                new TrainerEarlyStoppingMetadata.MonitorState(1, 0.4, 0.5, 2),
                true,
                4);

        assertEquals(Boolean.FALSE, metadata.get("earlyStoppingEnabled"));
        assertEquals(0, metadata.get("earlyStoppingPatience"));
        assertEquals(0.0, metadata.get("earlyStoppingMinDelta"));
        assertEquals(Boolean.TRUE, metadata.get("earlyStoppingTriggered"));
        assertEquals(4, metadata.get("earlyStoppingEpoch"));
        assertEquals("early-stopping", metadata.get("stopReason"));
    }
}
