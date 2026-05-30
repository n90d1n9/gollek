package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerMetricStateTest {
    @Test
    void snapshotsAreCopiedAndResetTogether() {
        TrainerMetricState state = new TrainerMetricState();
        Map<String, Double> trainValues = new LinkedHashMap<>();
        trainValues.put("mae", 0.25);
        Map<String, Object> trainDetails = new LinkedHashMap<>();
        trainDetails.put("count", 4);

        state.recordTrain(new TrainerMetricRuntime.MetricSnapshot(trainValues, trainDetails));
        trainValues.put("mae", 9.0);
        trainDetails.put("count", 99);

        assertEquals(0.25, state.trainValues().get("mae"), 1e-9);
        assertEquals(4, state.trainDetails().get("count"));
        assertThrows(UnsupportedOperationException.class, () -> state.trainValues().put("new", 1.0));

        state.recordValidation(new TrainerMetricRuntime.MetricSnapshot(
                Map.of("accuracy", 0.75),
                Map.of("tp", 3)));

        assertEquals(0.75, state.validationValues().get("accuracy"), 1e-9);
        assertEquals(3, state.validationDetails().get("tp"));

        state.reset();

        assertTrue(state.trainValues().isEmpty());
        assertTrue(state.validationValues().isEmpty());
        assertTrue(state.trainDetails().isEmpty());
        assertTrue(state.validationDetails().isEmpty());
    }
}
