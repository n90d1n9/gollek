package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.function.BiConsumer;

/**
 * Compatibility guards for trainer-owned checkpoint metadata.
 */
final class TrainerCheckpointMetadataGuards {
    private TrainerCheckpointMetadataGuards() {
    }

    static void requireMetadataMatch(
            String artifactName,
            Map<String, Object> state,
            String key,
            String expected,
            String label,
            BiConsumer<String, String> mismatchRecorder) {
        Object raw = state.get(key);
        if (raw == null) {
            return;
        }
        String actual = String.valueOf(raw);
        if (expected.equals(actual)) {
            return;
        }
        String error = label + " mismatch (expected " + expected + ", got " + actual + ")";
        mismatchRecorder.accept(artifactName, error);
        throw new IllegalArgumentException("Checkpoint metadata mismatch: " + error);
    }

    static void requireSchedulerStepUnit(Map<String, Object> state, String expectedStepUnit) {
        Object rawStepUnit = state.get("trainerSchedulerStepUnit");
        if (rawStepUnit == null) {
            return;
        }
        String loadedStepUnit = String.valueOf(rawStepUnit);
        if (!expectedStepUnit.equals(loadedStepUnit)) {
            throw new IllegalArgumentException(
                    "Invalid scheduler checkpoint payload: step unit mismatch (expected "
                            + expectedStepUnit + ", got " + loadedStepUnit + ")");
        }
    }

    static void requireMixedPrecisionEnabled(Map<String, Object> state) {
        Object enabled = state.get("trainerMixedPrecisionEnabled");
        if (enabled == null) {
            return;
        }
        if (enabled instanceof Boolean flag && flag) {
            return;
        }
        if (enabled instanceof String text && "true".equalsIgnoreCase(text)) {
            return;
        }
        throw new IllegalArgumentException("Invalid GradScaler checkpoint payload: mixed precision was not enabled");
    }
}
