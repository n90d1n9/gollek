package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.Acceleration;

class TrainerAccelerationMetadataTest {

    @Test
    void publishesCurrentBackendStatusAndMatmulDelta() {
        Map<String, Object> metadata = new HashMap<>();
        Acceleration.BackendStatus start = new Acceleration.BackendStatus(
                "metal",
                "Apple GPU",
                true,
                true,
                10);
        Acceleration.BackendStatus current = new Acceleration.BackendStatus(
                "metal",
                "Apple GPU",
                true,
                true,
                17);

        TrainerAccelerationMetadata.put(metadata, "metal", current, start);

        assertEquals("metal", metadata.get("requestedDevice"));
        assertEquals("metal", metadata.get("executionBackend"));
        assertEquals("Apple GPU", metadata.get("executionDeviceName"));
        assertEquals(Boolean.TRUE, metadata.get("executionAccelerated"));
        assertEquals(Boolean.TRUE, metadata.get("requestedDeviceAvailable"));
        assertEquals(17L, metadata.get("acceleratedMatmulCalls"));
        assertEquals(10L, metadata.get("acceleratedMatmulCallsAtStart"));
        assertEquals(7L, metadata.get("acceleratedMatmulCallsDelta"));
    }

    @Test
    void clampsMatmulDeltaWhenCountersMoveBackwards() {
        Map<String, Object> metadata = new HashMap<>();
        Acceleration.BackendStatus start = new Acceleration.BackendStatus("metal", "Apple GPU", true, true, 20);
        Acceleration.BackendStatus current = new Acceleration.BackendStatus("cpu", "CPU", false, true, 12);

        TrainerAccelerationMetadata.put(metadata, "auto", current, start);

        assertEquals(0L, metadata.get("acceleratedMatmulCallsDelta"));
    }

    @Test
    void omitsStartOnlyFieldsWhenStartStatusIsUnknown() {
        Map<String, Object> metadata = new HashMap<>();
        Acceleration.BackendStatus current = new Acceleration.BackendStatus("cpu", "CPU", false, true, 12);

        TrainerAccelerationMetadata.put(metadata, "cpu", current, null);

        assertEquals(12L, metadata.get("acceleratedMatmulCalls"));
        assertFalse(metadata.containsKey("acceleratedMatmulCallsAtStart"));
        assertFalse(metadata.containsKey("acceleratedMatmulCallsDelta"));
    }
}
