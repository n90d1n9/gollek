package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerCheckpointStatePayloadsTest {

    @Test
    void optimizerPayloadCopiesStateAndAddsTrainerMetadata() {
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("momentum", 0.9);
        componentState.put("trainerOptimizerClass", "stale");

        Map<String, Object> payload = TrainerCheckpointStatePayloads.optimizer(
                componentState,
                7,
                "SgdOptimizer",
                "weight:[2]:2;");

        assertEquals(0.9, payload.get("momentum"));
        assertEquals(7, payload.get("trainerOptimizerStepCount"));
        assertEquals("SgdOptimizer", payload.get("trainerOptimizerClass"));
        assertEquals("weight:[2]:2;", payload.get("trainerOptimizerParameterSignature"));
        assertEquals("stale", componentState.get("trainerOptimizerClass"));
    }

    @Test
    void schedulerPayloadHandlesNullStateAndAddsTrainerMetadata() {
        Map<String, Object> payload = TrainerCheckpointStatePayloads.scheduler(
                null,
                "StepLR",
                "EPOCH",
                3,
                "0:[1]:1;");

        assertEquals(4, payload.size());
        assertEquals("StepLR", payload.get("trainerSchedulerClass"));
        assertEquals("EPOCH", payload.get("trainerSchedulerStepUnit"));
        assertEquals(3, payload.get("trainerSchedulerStepCount"));
        assertEquals("0:[1]:1;", payload.get("trainerSchedulerOptimizerParameterSignature"));
    }

    @Test
    void gradScalerPayloadMarksMixedPrecisionEnabledAndOverridesStaleKeys() {
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("scale", 128.0);
        componentState.put("trainerMixedPrecisionEnabled", false);

        Map<String, Object> payload = TrainerCheckpointStatePayloads.gradScaler(
                componentState,
                "GradScaler",
                2);

        assertEquals(128.0, payload.get("scale"));
        assertEquals("GradScaler", payload.get("trainerGradScalerClass"));
        assertEquals(Boolean.TRUE, payload.get("trainerMixedPrecisionEnabled"));
        assertEquals(2, payload.get("trainerMixedPrecisionOverflowSkipCount"));
        assertEquals(false, componentState.get("trainerMixedPrecisionEnabled"));
    }

    @Test
    void returnedPayloadIsIndependentFromSourceMapMutations() {
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("scale", 128.0);

        Map<String, Object> payload = TrainerCheckpointStatePayloads.gradScaler(componentState, "GradScaler", 0);
        componentState.put("later", "mutation");

        assertFalse(payload.containsKey("later"));
    }

    @Test
    void restoredCountersParseStringsAndClampNegativeValues() {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("trainerOptimizerStepCount", "12");
        state.put("trainerSchedulerStepCount", -7);
        state.put("trainerMixedPrecisionOverflowSkipCount", 3L);

        assertEquals(12, TrainerCheckpointStatePayloads.optimizerStepCount(state, 1));
        assertEquals(0, TrainerCheckpointStatePayloads.schedulerStepCount(state, 5));
        assertEquals(3, TrainerCheckpointStatePayloads.mixedPrecisionOverflowSkipCount(state, 0));
    }

    @Test
    void restoredCountersUseFallbackForMissingOrInvalidState() {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("trainerOptimizerStepCount", "not-a-number");

        assertEquals(8, TrainerCheckpointStatePayloads.optimizerStepCount(state, 8));
        assertEquals(4, TrainerCheckpointStatePayloads.schedulerStepCount(state, 4));
        assertEquals(2, TrainerCheckpointStatePayloads.mixedPrecisionOverflowSkipCount(null, 2));
    }
}
