package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Backend-neutral diagnostics for a generic token dataset.
 */
public record DiscreteTokenDatasetProfile(
        int exampleCount,
        Map<String, Integer> taskExampleCounts,
        DiscreteTokenLengthStats inputLengths,
        DiscreteTokenLengthStats targetLengths,
        long observedTokenCount,
        int distinctTokenCount,
        int minToken,
        int maxToken,
        int knownSolutionExampleCount,
        int unknownSolutionExampleCount,
        int minKnownSolutionCount,
        int maxKnownSolutionCount) {

    public DiscreteTokenDatasetProfile {
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
        taskExampleCounts = Map.copyOf(Objects.requireNonNull(taskExampleCounts, "taskExampleCounts must not be null"));
        int taskTotal = 0;
        for (Map.Entry<String, Integer> entry : taskExampleCounts.entrySet()) {
            if (entry.getKey() == null || entry.getKey().isBlank()) {
                throw new IllegalArgumentException("taskExampleCounts keys must not be null or blank");
            }
            int count = Objects.requireNonNull(entry.getValue(), "taskExampleCounts values must not be null");
            if (count <= 0) {
                throw new IllegalArgumentException("taskExampleCounts values must be > 0 but was " + count);
            }
            taskTotal += count;
        }
        if (taskTotal != exampleCount) {
            throw new IllegalArgumentException("sum of taskExampleCounts must equal exampleCount");
        }

        inputLengths = Objects.requireNonNull(inputLengths, "inputLengths must not be null");
        targetLengths = Objects.requireNonNull(targetLengths, "targetLengths must not be null");
        if (inputLengths.sequenceCount() != exampleCount) {
            throw new IllegalArgumentException("inputLengths sequence count must equal exampleCount");
        }
        if (targetLengths.sequenceCount() != exampleCount) {
            throw new IllegalArgumentException("targetLengths sequence count must equal exampleCount");
        }
        long expectedObservedTokenCount = inputLengths.totalLength() + targetLengths.totalLength();
        if (observedTokenCount != expectedObservedTokenCount) {
            throw new IllegalArgumentException("observedTokenCount must equal input + target token totals");
        }
        if (distinctTokenCount < 0) {
            throw new IllegalArgumentException("distinctTokenCount must be >= 0 but was " + distinctTokenCount);
        }
        if (distinctTokenCount > observedTokenCount) {
            throw new IllegalArgumentException("distinctTokenCount cannot exceed observedTokenCount");
        }
        if (exampleCount == 0) {
            if (observedTokenCount != 0
                    || distinctTokenCount != 0
                    || minToken != 0
                    || maxToken != 0) {
                throw new IllegalArgumentException("empty dataset profile must use zero token range/counts");
            }
        } else if (minToken > maxToken) {
            throw new IllegalArgumentException("minToken must be <= maxToken");
        }

        if (knownSolutionExampleCount < 0) {
            throw new IllegalArgumentException(
                    "knownSolutionExampleCount must be >= 0 but was " + knownSolutionExampleCount);
        }
        if (unknownSolutionExampleCount < 0) {
            throw new IllegalArgumentException(
                    "unknownSolutionExampleCount must be >= 0 but was " + unknownSolutionExampleCount);
        }
        if (knownSolutionExampleCount + unknownSolutionExampleCount != exampleCount) {
            throw new IllegalArgumentException(
                    "knownSolutionExampleCount + unknownSolutionExampleCount must equal exampleCount");
        }
        if (knownSolutionExampleCount == 0) {
            if (minKnownSolutionCount != -1 || maxKnownSolutionCount != -1) {
                throw new IllegalArgumentException("unknown-only profiles must use -1 known-solution min/max");
            }
        } else {
            if (minKnownSolutionCount < 1 || maxKnownSolutionCount < minKnownSolutionCount) {
                throw new IllegalArgumentException("known-solution min/max must be positive and ordered");
            }
        }
    }

    public int taskCount() {
        return taskExampleCounts.size();
    }

    public boolean hasExamples() {
        return exampleCount > 0;
    }

    public boolean hasKnownSolutionCounts() {
        return knownSolutionExampleCount > 0;
    }

    public double knownSolutionCoverageRate() {
        return exampleCount == 0 ? 0.0d : (double) knownSolutionExampleCount / exampleCount;
    }
}
