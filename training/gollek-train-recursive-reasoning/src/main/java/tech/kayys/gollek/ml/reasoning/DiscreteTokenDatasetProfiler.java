package tech.kayys.gollek.ml.reasoning;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Builds compact diagnostics for generic token datasets.
 */
public final class DiscreteTokenDatasetProfiler {
    private DiscreteTokenDatasetProfiler() {}

    public static DiscreteTokenDatasetProfile profile(List<DiscreteTokenDatasetExample> examples) {
        Objects.requireNonNull(examples, "examples must not be null");
        if (examples.isEmpty()) {
            return new DiscreteTokenDatasetProfile(
                    0,
                    Map.of(),
                    DiscreteTokenLengthStats.empty(),
                    DiscreteTokenLengthStats.empty(),
                    0L,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    -1);
        }

        Map<String, Integer> taskCounts = new LinkedHashMap<>();
        Set<Integer> distinctTokens = new TreeSet<>();
        int minInputLength = Integer.MAX_VALUE;
        int maxInputLength = 0;
        long totalInputLength = 0L;
        int minTargetLength = Integer.MAX_VALUE;
        int maxTargetLength = 0;
        long totalTargetLength = 0L;
        int minToken = Integer.MAX_VALUE;
        int maxToken = Integer.MIN_VALUE;
        int knownSolutionExamples = 0;
        int minKnownSolutionCount = Integer.MAX_VALUE;
        int maxKnownSolutionCount = 0;

        for (int index = 0; index < examples.size(); index++) {
            DiscreteTokenDatasetExample example =
                    Objects.requireNonNull(examples.get(index), "examples[" + index + "] must not be null");
            taskCounts.merge(example.taskId(), 1, Integer::sum);

            int[] input = example.inputTokens();
            int[] target = example.targetTokens();
            minInputLength = Math.min(minInputLength, input.length);
            maxInputLength = Math.max(maxInputLength, input.length);
            totalInputLength += input.length;
            minTargetLength = Math.min(minTargetLength, target.length);
            maxTargetLength = Math.max(maxTargetLength, target.length);
            totalTargetLength += target.length;

            for (int token : input) {
                distinctTokens.add(token);
                minToken = Math.min(minToken, token);
                maxToken = Math.max(maxToken, token);
            }
            for (int token : target) {
                distinctTokens.add(token);
                minToken = Math.min(minToken, token);
                maxToken = Math.max(maxToken, token);
            }

            if (example.hasKnownSolutionCount()) {
                knownSolutionExamples++;
                minKnownSolutionCount = Math.min(minKnownSolutionCount, example.knownSolutionCount());
                maxKnownSolutionCount = Math.max(maxKnownSolutionCount, example.knownSolutionCount());
            }
        }

        int exampleCount = examples.size();
        return new DiscreteTokenDatasetProfile(
                exampleCount,
                taskCounts,
                DiscreteTokenLengthStats.of(exampleCount, minInputLength, maxInputLength, totalInputLength),
                DiscreteTokenLengthStats.of(exampleCount, minTargetLength, maxTargetLength, totalTargetLength),
                totalInputLength + totalTargetLength,
                distinctTokens.size(),
                minToken,
                maxToken,
                knownSolutionExamples,
                exampleCount - knownSolutionExamples,
                knownSolutionExamples == 0 ? -1 : minKnownSolutionCount,
                knownSolutionExamples == 0 ? -1 : maxKnownSolutionCount);
    }
}
