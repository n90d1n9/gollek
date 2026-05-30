package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Warning policy for dataset-plan diagnostics.
 */
public record DiscreteTokenDatasetPlanDiagnosticsPolicy(
        double highPaddingRateThreshold,
        boolean warnOnMissingValidationSplit,
        boolean warnOnMissingTestSplit,
        boolean warnOnDroppedTrainingExamples,
        boolean warnOnMissingKnownSolutionCounts,
        boolean warnOnPartialKnownSolutionCoverage,
        boolean warnOnHighPaddingRate,
        boolean warnOnMissingTaskTrainCoverage,
        boolean warnOnMissingTaskValidationCoverage,
        boolean warnOnMissingTaskTestCoverage) {

    public DiscreteTokenDatasetPlanDiagnosticsPolicy {
        if (!isRate(highPaddingRateThreshold)) {
            throw new IllegalArgumentException(
                    "highPaddingRateThreshold must be finite and in [0, 1] but was "
                            + highPaddingRateThreshold);
        }
    }

    public static DiscreteTokenDatasetPlanDiagnosticsPolicy defaults() {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                DiscreteTokenDatasetPlanDiagnostics.DEFAULT_HIGH_PADDING_RATE_THRESHOLD,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true);
    }

    public static DiscreteTokenDatasetPlanDiagnosticsPolicy lenient() {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                DiscreteTokenDatasetPlanDiagnostics.DEFAULT_HIGH_PADDING_RATE_THRESHOLD,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false);
    }

    public static DiscreteTokenDatasetPlanDiagnosticsPolicy fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                requiredDouble(metadata, "highPaddingRateThreshold"),
                requiredBoolean(metadata, "warnOnMissingValidationSplit"),
                requiredBoolean(metadata, "warnOnMissingTestSplit"),
                requiredBoolean(metadata, "warnOnDroppedTrainingExamples"),
                requiredBoolean(metadata, "warnOnMissingKnownSolutionCounts"),
                requiredBoolean(metadata, "warnOnPartialKnownSolutionCoverage"),
                requiredBoolean(metadata, "warnOnHighPaddingRate"),
                requiredBoolean(metadata, "warnOnMissingTaskTrainCoverage"),
                requiredBoolean(metadata, "warnOnMissingTaskValidationCoverage"),
                requiredBoolean(metadata, "warnOnMissingTaskTestCoverage"));
    }

    public DiscreteTokenDatasetPlanDiagnosticsPolicy withHighPaddingRateThreshold(double threshold) {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                threshold,
                warnOnMissingValidationSplit,
                warnOnMissingTestSplit,
                warnOnDroppedTrainingExamples,
                warnOnMissingKnownSolutionCounts,
                warnOnPartialKnownSolutionCoverage,
                warnOnHighPaddingRate,
                warnOnMissingTaskTrainCoverage,
                warnOnMissingTaskValidationCoverage,
                warnOnMissingTaskTestCoverage);
    }

    public DiscreteTokenDatasetPlanDiagnosticsPolicy withEvaluationSplitWarnings(boolean enabled) {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                highPaddingRateThreshold,
                enabled,
                enabled,
                warnOnDroppedTrainingExamples,
                warnOnMissingKnownSolutionCounts,
                warnOnPartialKnownSolutionCoverage,
                warnOnHighPaddingRate,
                warnOnMissingTaskTrainCoverage,
                enabled,
                enabled);
    }

    public DiscreteTokenDatasetPlanDiagnosticsPolicy withKnownSolutionWarnings(boolean enabled) {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                highPaddingRateThreshold,
                warnOnMissingValidationSplit,
                warnOnMissingTestSplit,
                warnOnDroppedTrainingExamples,
                enabled,
                enabled,
                warnOnHighPaddingRate,
                warnOnMissingTaskTrainCoverage,
                warnOnMissingTaskValidationCoverage,
                warnOnMissingTaskTestCoverage);
    }

    public DiscreteTokenDatasetPlanDiagnosticsPolicy withPerTaskSplitWarnings(boolean enabled) {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                highPaddingRateThreshold,
                warnOnMissingValidationSplit,
                warnOnMissingTestSplit,
                warnOnDroppedTrainingExamples,
                warnOnMissingKnownSolutionCounts,
                warnOnPartialKnownSolutionCoverage,
                warnOnHighPaddingRate,
                enabled,
                enabled,
                enabled);
    }

    public DiscreteTokenDatasetPlanDiagnosticsPolicy withPaddingWarning(boolean enabled) {
        return new DiscreteTokenDatasetPlanDiagnosticsPolicy(
                highPaddingRateThreshold,
                warnOnMissingValidationSplit,
                warnOnMissingTestSplit,
                warnOnDroppedTrainingExamples,
                warnOnMissingKnownSolutionCounts,
                warnOnPartialKnownSolutionCoverage,
                enabled,
                warnOnMissingTaskTrainCoverage,
                warnOnMissingTaskValidationCoverage,
                warnOnMissingTaskTestCoverage);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("highPaddingRateThreshold", highPaddingRateThreshold);
        metadata.put("warnOnMissingValidationSplit", warnOnMissingValidationSplit);
        metadata.put("warnOnMissingTestSplit", warnOnMissingTestSplit);
        metadata.put("warnOnDroppedTrainingExamples", warnOnDroppedTrainingExamples);
        metadata.put("warnOnMissingKnownSolutionCounts", warnOnMissingKnownSolutionCounts);
        metadata.put("warnOnPartialKnownSolutionCoverage", warnOnPartialKnownSolutionCoverage);
        metadata.put("warnOnHighPaddingRate", warnOnHighPaddingRate);
        metadata.put("warnOnMissingTaskTrainCoverage", warnOnMissingTaskTrainCoverage);
        metadata.put("warnOnMissingTaskValidationCoverage", warnOnMissingTaskValidationCoverage);
        metadata.put("warnOnMissingTaskTestCoverage", warnOnMissingTaskTestCoverage);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static double requiredDouble(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Double.parseDouble(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be a number", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a number");
    }

    private static boolean requiredBoolean(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof CharSequence text) {
            String normalized = text.toString().trim().toLowerCase();
            if ("true".equals(normalized)) {
                return true;
            }
            if ("false".equals(normalized)) {
                return false;
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a boolean");
    }

    private static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }

    private static boolean isRate(double value) {
        return Double.isFinite(value) && value >= 0.0d && value <= 1.0d;
    }
}
