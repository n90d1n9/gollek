package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Pads benchmark-neutral token examples into trainer-facing batches.
 */
public final class DiscreteTokenDatasetBatcher {
    private DiscreteTokenDatasetBatcher() {}

    public static DiscreteTokenDatasetBatch batch(List<DiscreteTokenDatasetExample> examples, int padToken) {
        return batch(examples, padToken, padToken);
    }

    public static DiscreteTokenDatasetBatch batch(
            List<DiscreteTokenDatasetExample> examples,
            int inputPadToken,
            int targetPadToken) {
        Objects.requireNonNull(examples, "examples must not be null");
        if (examples.isEmpty()) {
            throw new IllegalArgumentException("examples must not be empty");
        }

        int batchSize = examples.size();
        int maxInputLength = 0;
        int maxTargetLength = 0;
        for (int row = 0; row < batchSize; row++) {
            DiscreteTokenDatasetExample example =
                    Objects.requireNonNull(examples.get(row), "examples[" + row + "] must not be null");
            maxInputLength = Math.max(maxInputLength, example.inputTokenCount());
            maxTargetLength = Math.max(maxTargetLength, example.targetTokenCount());
        }

        String[] taskIds = new String[batchSize];
        int[] exampleIndices = new int[batchSize];
        int[][] inputTokens = new int[batchSize][maxInputLength];
        int[][] targetTokens = new int[batchSize][maxTargetLength];
        int[][] inputMask = new int[batchSize][maxInputLength];
        int[][] targetMask = new int[batchSize][maxTargetLength];
        int[] inputLengths = new int[batchSize];
        int[] targetLengths = new int[batchSize];
        int[] knownSolutionCounts = new int[batchSize];
        @SuppressWarnings("unchecked")
        Map<String, Object>[] metadata = new Map[batchSize];

        for (int row = 0; row < batchSize; row++) {
            DiscreteTokenDatasetExample example = examples.get(row);
            int[] input = example.inputTokens();
            int[] target = example.targetTokens();

            Arrays.fill(inputTokens[row], inputPadToken);
            Arrays.fill(targetTokens[row], targetPadToken);
            System.arraycopy(input, 0, inputTokens[row], 0, input.length);
            System.arraycopy(target, 0, targetTokens[row], 0, target.length);
            Arrays.fill(inputMask[row], 0, input.length, 1);
            Arrays.fill(targetMask[row], 0, target.length, 1);

            taskIds[row] = example.taskId();
            exampleIndices[row] = example.exampleIndex();
            inputLengths[row] = input.length;
            targetLengths[row] = target.length;
            knownSolutionCounts[row] = example.knownSolutionCount();
            metadata[row] = example.metadata();
        }

        return new DiscreteTokenDatasetBatch(
                taskIds,
                exampleIndices,
                inputTokens,
                targetTokens,
                inputMask,
                targetMask,
                inputLengths,
                targetLengths,
                knownSolutionCounts,
                List.of(metadata),
                inputPadToken,
                targetPadToken);
    }
}
