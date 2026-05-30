package tech.kayys.gollek.ml.reasoning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * One N-Queens training/evaluation example: partial board input plus one valid target solution.
 */
public record NQueensDatasetExample(
        int exampleIndex,
        NQueensProblem problem,
        NQueensSolution targetSolution,
        int knownSolutionCount,
        Map<String, Object> metadata) {

    public NQueensDatasetExample {
        if (exampleIndex < 0) {
            throw new IllegalArgumentException("exampleIndex must be >= 0 but was " + exampleIndex);
        }
        problem = Objects.requireNonNull(problem, "problem must not be null");
        targetSolution = Objects.requireNonNull(targetSolution, "targetSolution must not be null");
        if (problem.size() != targetSolution.size()) {
            throw new IllegalArgumentException(
                    "targetSolution size must match problem size: "
                            + targetSolution.size()
                            + " vs "
                            + problem.size());
        }
        if (!NQueensBenchmark.evaluate(problem, targetSolution).valid()) {
            throw new IllegalArgumentException("targetSolution must be valid for problem");
        }
        if (knownSolutionCount < 1) {
            throw new IllegalArgumentException("knownSolutionCount must be >= 1 but was " + knownSolutionCount);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public int fixedQueenCount() {
        int count = 0;
        for (int row = 0; row < problem.size(); row++) {
            if (problem.hasFixedQueen(row)) {
                count++;
            }
        }
        return count;
    }

    public int removedQueenCount() {
        return problem.size() - fixedQueenCount();
    }

    public int[] inputTokens() {
        return problem.toTokens();
    }

    public int[] targetTokens() {
        return targetSolution.toTokens();
    }

    public DiscreteTokenDatasetExample toTokenExample() {
        Map<String, Object> tokenMetadata = new LinkedHashMap<>(metadata);
        tokenMetadata.put("taskType", "nqueens");
        tokenMetadata.put("size", problem.size());
        tokenMetadata.put("fixedQueenCount", fixedQueenCount());
        tokenMetadata.put("removedQueenCount", removedQueenCount());
        return new DiscreteTokenDatasetExample(
                "nqueens",
                exampleIndex,
                inputTokens(),
                targetTokens(),
                knownSolutionCount,
                tokenMetadata);
    }
}
