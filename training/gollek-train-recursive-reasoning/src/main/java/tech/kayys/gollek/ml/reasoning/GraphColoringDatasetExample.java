package tech.kayys.gollek.ml.reasoning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * One Graph Coloring training/evaluation example: partial colors plus one target solution.
 */
public record GraphColoringDatasetExample(
        int exampleIndex,
        GraphColoringProblem problem,
        GraphColoringSolution targetSolution,
        int knownSolutionCount,
        Map<String, Object> metadata) {

    public GraphColoringDatasetExample {
        if (exampleIndex < 0) {
            throw new IllegalArgumentException("exampleIndex must be >= 0 but was " + exampleIndex);
        }
        problem = Objects.requireNonNull(problem, "problem must not be null");
        targetSolution = Objects.requireNonNull(targetSolution, "targetSolution must not be null");
        if (problem.nodeCount() != targetSolution.nodeCount()) {
            throw new IllegalArgumentException(
                    "targetSolution node count must match problem node count: "
                            + targetSolution.nodeCount() + " vs " + problem.nodeCount());
        }
        if (problem.colorCount() != targetSolution.colorCount()) {
            throw new IllegalArgumentException(
                    "targetSolution color count must match problem color count: "
                            + targetSolution.colorCount() + " vs " + problem.colorCount());
        }
        if (!GraphColoringBenchmark.evaluate(problem, targetSolution).valid()) {
            throw new IllegalArgumentException("targetSolution must be valid for problem");
        }
        if (knownSolutionCount < 1) {
            throw new IllegalArgumentException("knownSolutionCount must be >= 1 but was " + knownSolutionCount);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public int fixedColorCount() {
        int count = 0;
        for (int node = 0; node < problem.nodeCount(); node++) {
            if (problem.hasFixedColor(node)) {
                count++;
            }
        }
        return count;
    }

    public int removedColorCount() {
        return problem.nodeCount() - fixedColorCount();
    }

    public int[] inputTokens() {
        return problem.toTokens();
    }

    public int[] targetTokens() {
        return targetSolution.toTokens();
    }

    public DiscreteTokenDatasetExample toTokenExample() {
        Map<String, Object> tokenMetadata = new LinkedHashMap<>(metadata);
        tokenMetadata.put("taskType", "graph-coloring");
        tokenMetadata.put("nodeCount", problem.nodeCount());
        tokenMetadata.put("colorCount", problem.colorCount());
        tokenMetadata.put("edgeCount", problem.edges().size());
        tokenMetadata.put("fixedColorCount", fixedColorCount());
        tokenMetadata.put("removedColorCount", removedColorCount());
        return new DiscreteTokenDatasetExample(
                "graph-coloring",
                exampleIndex,
                inputTokens(),
                targetTokens(),
                knownSolutionCount,
                tokenMetadata);
    }
}
