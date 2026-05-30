package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Benchmark-neutral token-pair example for structured reasoning trainers.
 */
public record DiscreteTokenDatasetExample(
        String taskId,
        int exampleIndex,
        int[] inputTokens,
        int[] targetTokens,
        int knownSolutionCount,
        Map<String, Object> metadata) {

    public DiscreteTokenDatasetExample {
        taskId = Objects.requireNonNull(taskId, "taskId must not be null");
        if (taskId.isBlank()) {
            throw new IllegalArgumentException("taskId must not be blank");
        }
        if (exampleIndex < 0) {
            throw new IllegalArgumentException("exampleIndex must be >= 0 but was " + exampleIndex);
        }
        inputTokens = Objects.requireNonNull(inputTokens, "inputTokens must not be null").clone();
        targetTokens = Objects.requireNonNull(targetTokens, "targetTokens must not be null").clone();
        if (inputTokens.length == 0) {
            throw new IllegalArgumentException("inputTokens must not be empty");
        }
        if (targetTokens.length == 0) {
            throw new IllegalArgumentException("targetTokens must not be empty");
        }
        if (knownSolutionCount < -1 || knownSolutionCount == 0) {
            throw new IllegalArgumentException(
                    "knownSolutionCount must be -1 or >= 1 but was " + knownSolutionCount);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] inputTokens() {
        return inputTokens.clone();
    }

    @Override
    public int[] targetTokens() {
        return targetTokens.clone();
    }

    public boolean hasKnownSolutionCount() {
        return knownSolutionCount > 0;
    }

    public int inputTokenCount() {
        return inputTokens.length;
    }

    public int targetTokenCount() {
        return targetTokens.length;
    }
}
