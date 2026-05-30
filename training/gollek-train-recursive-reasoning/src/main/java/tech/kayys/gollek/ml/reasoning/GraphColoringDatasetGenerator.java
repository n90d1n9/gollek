package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Deterministic Graph Coloring dataset generator backed by exact solver completions.
 */
public final class GraphColoringDatasetGenerator {
    private GraphColoringDatasetGenerator() {
    }

    public static List<GraphColoringDatasetExample> generate(
            GraphColoringProblem sourceProblem,
            int exampleCount,
            int fixedColorCount,
            long seed) {
        Objects.requireNonNull(sourceProblem, "sourceProblem must not be null");
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
        int sourceFixedColorCount = fixedColorCount(sourceProblem);
        if (fixedColorCount < sourceFixedColorCount || fixedColorCount > sourceProblem.nodeCount()) {
            throw new IllegalArgumentException(
                    "fixedColorCount must be in ["
                            + sourceFixedColorCount
                            + ", "
                            + sourceProblem.nodeCount()
                            + "] but was "
                            + fixedColorCount);
        }
        if (exampleCount == 0) {
            return List.of();
        }

        List<GraphColoringSolution> sourceSolutions = GraphColoringSolver.solve(sourceProblem);
        if (sourceSolutions.isEmpty()) {
            throw new IllegalArgumentException("sourceProblem has no complete Graph Coloring solutions");
        }

        Random random = new Random(seed);
        List<GraphColoringDatasetExample> examples = new ArrayList<>(exampleCount);
        for (int exampleIndex = 0; exampleIndex < exampleCount; exampleIndex++) {
            int solutionIndex = random.nextInt(sourceSolutions.size());
            GraphColoringSolution target = sourceSolutions.get(solutionIndex);
            GraphColoringProblem maskedProblem = maskTarget(sourceProblem, target, fixedColorCount, random);
            int knownSolutionCount = GraphColoringSolver.count(maskedProblem);
            examples.add(new GraphColoringDatasetExample(
                    exampleIndex,
                    maskedProblem,
                    target,
                    knownSolutionCount,
                    Map.of(
                            "seed", seed,
                            "sourceFixedColorCount", sourceFixedColorCount,
                            "targetSolutionIndex", solutionIndex,
                            "sourceSolutionCount", sourceSolutions.size())));
        }
        return List.copyOf(examples);
    }

    private static GraphColoringProblem maskTarget(
            GraphColoringProblem sourceProblem,
            GraphColoringSolution target,
            int fixedColorCount,
            Random random) {
        int[] fixedColors = new int[sourceProblem.nodeCount()];
        java.util.Arrays.fill(fixedColors, GraphColoringProblem.UNCOLORED);

        List<Integer> nodesToReveal = new ArrayList<>();
        for (int node = 0; node < sourceProblem.nodeCount(); node++) {
            if (sourceProblem.hasFixedColor(node)) {
                fixedColors[node] = target.color(node);
            } else {
                nodesToReveal.add(node);
            }
        }

        Collections.shuffle(nodesToReveal, random);
        int nodesNeeded = fixedColorCount - fixedColorCount(sourceProblem);
        for (int i = 0; i < nodesNeeded; i++) {
            int node = nodesToReveal.get(i);
            fixedColors[node] = target.color(node);
        }
        return new GraphColoringProblem(
                sourceProblem.nodeCount(),
                sourceProblem.colorCount(),
                sourceProblem.edges(),
                fixedColors);
    }

    private static int fixedColorCount(GraphColoringProblem problem) {
        int count = 0;
        for (int node = 0; node < problem.nodeCount(); node++) {
            if (problem.hasFixedColor(node)) {
                count++;
            }
        }
        return count;
    }
}
