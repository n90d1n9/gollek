package tech.kayys.gollek.ml.train;

import java.util.HashMap;
import java.util.Map;

/**
 * Builds trainer-owned metadata payloads layered onto component state dictionaries.
 */
final class TrainerCheckpointStatePayloads {
    private TrainerCheckpointStatePayloads() {
    }

    static Map<String, Object> optimizer(
            Map<String, Object> optimizerState,
            int optimizerStepCount,
            String optimizerClass,
            String parameterSignature) {
        Map<String, Object> state = copyState(optimizerState);
        state.put("trainerOptimizerStepCount", optimizerStepCount);
        state.put("trainerOptimizerClass", optimizerClass);
        state.put("trainerOptimizerParameterSignature", parameterSignature);
        return state;
    }

    static Map<String, Object> scheduler(
            Map<String, Object> schedulerState,
            String schedulerClass,
            String stepUnit,
            int schedulerStepCount,
            String optimizerParameterSignature) {
        Map<String, Object> state = copyState(schedulerState);
        state.put("trainerSchedulerClass", schedulerClass);
        state.put("trainerSchedulerStepUnit", stepUnit);
        state.put("trainerSchedulerStepCount", schedulerStepCount);
        state.put("trainerSchedulerOptimizerParameterSignature", optimizerParameterSignature);
        return state;
    }

    static Map<String, Object> gradScaler(
            Map<String, Object> gradScalerState,
            String gradScalerClass,
            int overflowSkipCount) {
        Map<String, Object> state = copyState(gradScalerState);
        state.put("trainerGradScalerClass", gradScalerClass);
        state.put("trainerMixedPrecisionEnabled", true);
        state.put("trainerMixedPrecisionOverflowSkipCount", overflowSkipCount);
        return state;
    }

    static int optimizerStepCount(Map<String, Object> state, int fallback) {
        return nonNegativeInt(state, "trainerOptimizerStepCount", fallback);
    }

    static int schedulerStepCount(Map<String, Object> state, int fallback) {
        return nonNegativeInt(state, "trainerSchedulerStepCount", fallback);
    }

    static int mixedPrecisionOverflowSkipCount(Map<String, Object> state, int fallback) {
        return nonNegativeInt(state, "trainerMixedPrecisionOverflowSkipCount", fallback);
    }

    private static Map<String, Object> copyState(Map<String, Object> state) {
        return new HashMap<>(state == null ? Map.of() : state);
    }

    private static int nonNegativeInt(Map<String, Object> state, String key, int fallback) {
        Object value = state == null ? null : state.get(key);
        return Math.max(0, TrainerMetadataSupport.readInt(value, fallback));
    }
}
