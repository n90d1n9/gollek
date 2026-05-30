package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Per-task train/validation/test split counts for dataset-plan diagnostics.
 */
public record DiscreteTokenDatasetTaskSplitDiagnostics(
        String taskId,
        int exampleCount,
        int trainCount,
        int validationCount,
        int testCount) {

    public DiscreteTokenDatasetTaskSplitDiagnostics {
        taskId = Objects.requireNonNull(taskId, "taskId must not be null");
        if (taskId.isBlank()) {
            throw new IllegalArgumentException("taskId must not be blank");
        }
        exampleCount = requireNonNegative(exampleCount, "exampleCount");
        trainCount = requireNonNegative(trainCount, "trainCount");
        validationCount = requireNonNegative(validationCount, "validationCount");
        testCount = requireNonNegative(testCount, "testCount");
        if (trainCount + validationCount + testCount != exampleCount) {
            throw new IllegalArgumentException("task split counts must sum to exampleCount");
        }
    }

    public static DiscreteTokenDatasetTaskSplitDiagnostics fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetTaskSplitDiagnostics(
                requiredString(metadata, "taskId"),
                requiredInt(metadata, "exampleCount"),
                requiredInt(metadata, "trainCount"),
                requiredInt(metadata, "validationCount"),
                requiredInt(metadata, "testCount"));
    }

    public boolean hasTrainExamples() {
        return trainCount > 0;
    }

    public boolean hasValidationExamples() {
        return validationCount > 0;
    }

    public boolean hasTestExamples() {
        return testCount > 0;
    }

    public double trainRate() {
        return exampleCount == 0 ? 0.0d : (double) trainCount / exampleCount;
    }

    public double validationRate() {
        return exampleCount == 0 ? 0.0d : (double) validationCount / exampleCount;
    }

    public double testRate() {
        return exampleCount == 0 ? 0.0d : (double) testCount / exampleCount;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("taskId", taskId);
        metadata.put("exampleCount", exampleCount);
        metadata.put("trainCount", trainCount);
        metadata.put("validationCount", validationCount);
        metadata.put("testCount", testCount);
        metadata.put("trainRate", trainRate());
        metadata.put("validationRate", validationRate());
        metadata.put("testRate", testRate());
        return Collections.unmodifiableMap(metadata);
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }

    private static String requiredString(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static int requiredInt(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            double numericValue = number.doubleValue();
            if (!Double.isFinite(numericValue)
                    || Math.rint(numericValue) != numericValue
                    || numericValue < Integer.MIN_VALUE
                    || numericValue > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
            }
            return number.intValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Integer.parseInt(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
    }

    private static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }
}
