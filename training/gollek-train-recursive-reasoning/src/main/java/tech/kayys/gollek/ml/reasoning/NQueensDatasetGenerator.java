package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Deterministic N-Queens dataset generator backed by exact solver completions.
 */
public final class NQueensDatasetGenerator {
    private NQueensDatasetGenerator() {
    }

    public static List<NQueensDatasetExample> generate(
            int size,
            int exampleCount,
            int fixedQueenCount,
            long seed) {
        return generate(NQueensProblem.empty(size), exampleCount, fixedQueenCount, seed);
    }

    public static List<NQueensDatasetExample> generate(
            NQueensProblem sourceProblem,
            int exampleCount,
            int fixedQueenCount,
            long seed) {
        Objects.requireNonNull(sourceProblem, "sourceProblem must not be null");
        if (exampleCount < 0) {
            throw new IllegalArgumentException("exampleCount must be >= 0 but was " + exampleCount);
        }
        int sourceFixedQueenCount = fixedQueenCount(sourceProblem);
        if (fixedQueenCount < sourceFixedQueenCount || fixedQueenCount > sourceProblem.size()) {
            throw new IllegalArgumentException(
                    "fixedQueenCount must be in ["
                            + sourceFixedQueenCount
                            + ", "
                            + sourceProblem.size()
                            + "] but was "
                            + fixedQueenCount);
        }
        if (exampleCount == 0) {
            return List.of();
        }

        List<NQueensSolution> sourceSolutions = NQueensSolver.solve(sourceProblem);
        if (sourceSolutions.isEmpty()) {
            throw new IllegalArgumentException("sourceProblem has no complete N-Queens solutions");
        }

        Random random = new Random(seed);
        List<NQueensDatasetExample> examples = new ArrayList<>(exampleCount);
        for (int exampleIndex = 0; exampleIndex < exampleCount; exampleIndex++) {
            int solutionIndex = random.nextInt(sourceSolutions.size());
            NQueensSolution target = sourceSolutions.get(solutionIndex);
            NQueensProblem maskedProblem = maskTarget(sourceProblem, target, fixedQueenCount, random);
            int knownSolutionCount = NQueensSolver.count(maskedProblem);
            examples.add(new NQueensDatasetExample(
                    exampleIndex,
                    maskedProblem,
                    target,
                    knownSolutionCount,
                    Map.of(
                            "seed", seed,
                            "sourceFixedQueenCount", sourceFixedQueenCount,
                            "targetSolutionIndex", solutionIndex,
                            "sourceSolutionCount", sourceSolutions.size())));
        }
        return List.copyOf(examples);
    }

    private static NQueensProblem maskTarget(
            NQueensProblem sourceProblem,
            NQueensSolution target,
            int fixedQueenCount,
            Random random) {
        int[] fixedColumns = new int[sourceProblem.size()];
        for (int row = 0; row < fixedColumns.length; row++) {
            fixedColumns[row] = NQueensProblem.EMPTY;
        }

        List<Integer> rowsToReveal = new ArrayList<>();
        for (int row = 0; row < sourceProblem.size(); row++) {
            if (sourceProblem.hasFixedQueen(row)) {
                fixedColumns[row] = target.column(row);
            } else {
                rowsToReveal.add(row);
            }
        }

        Collections.shuffle(rowsToReveal, random);
        int rowsNeeded = fixedQueenCount - fixedQueenCount(sourceProblem);
        for (int i = 0; i < rowsNeeded; i++) {
            int row = rowsToReveal.get(i);
            fixedColumns[row] = target.column(row);
        }
        return new NQueensProblem(sourceProblem.size(), fixedColumns);
    }

    private static int fixedQueenCount(NQueensProblem problem) {
        int count = 0;
        for (int row = 0; row < problem.size(); row++) {
            if (problem.hasFixedQueen(row)) {
                count++;
            }
        }
        return count;
    }
}
