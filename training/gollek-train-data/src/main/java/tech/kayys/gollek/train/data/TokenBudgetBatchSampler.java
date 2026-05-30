package tech.kayys.gollek.train.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Forms variable-size batches whose padded token capacity stays near a budget.
 *
 * <p>Batch cost is estimated as {@code maxSequenceLengthInBatch * batchSize}.
 * Samples longer than {@code maxTokens} are kept as single-sample batches because
 * the sampler cannot split one sequence.</p>
 */
public final class TokenBudgetBatchSampler implements BatchSampler {
    private final int[] lengths;
    private final int maxTokens;
    private final int maxExamples;
    private final boolean shuffleBatches;
    private final boolean shuffleWithinBatches;
    private final long seed;

    public TokenBudgetBatchSampler(int[] lengths, int maxTokens, long seed) {
        this(lengths, maxTokens, Integer.MAX_VALUE, true, false, seed);
    }

    public TokenBudgetBatchSampler(int[] lengths, int maxTokens, int maxExamples, long seed) {
        this(lengths, maxTokens, maxExamples, true, false, seed);
    }

    public TokenBudgetBatchSampler(
            int[] lengths,
            int maxTokens,
            int maxExamples,
            boolean shuffleBatches,
            boolean shuffleWithinBatches,
            long seed) {
        this.lengths = validateLengths(lengths);
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive, got: " + maxTokens);
        }
        if (maxExamples <= 0) {
            throw new IllegalArgumentException("maxExamples must be positive, got: " + maxExamples);
        }
        this.maxTokens = maxTokens;
        this.maxExamples = maxExamples;
        this.shuffleBatches = shuffleBatches;
        this.shuffleWithinBatches = shuffleWithinBatches;
        this.seed = seed;
    }

    @Override
    public List<List<Integer>> sampleBatches(int datasetSize) {
        requireDatasetSize(datasetSize);
        if (lengths.length == 0) {
            return List.of();
        }

        List<List<Integer>> batches = buildSortedBatches();
        Random random = new Random(seed);
        if (shuffleWithinBatches) {
            for (List<Integer> batch : batches) {
                Collections.shuffle(batch, random);
            }
        }
        if (shuffleBatches) {
            Collections.shuffle(batches, random);
        }

        List<List<Integer>> result = new ArrayList<>(batches.size());
        for (List<Integer> batch : batches) {
            result.add(List.copyOf(batch));
        }
        return List.copyOf(result);
    }

    @Override
    public int sampleCount(int datasetSize) {
        requireDatasetSize(datasetSize);
        return lengths.length;
    }

    @Override
    public int batchCount(int datasetSize) {
        return sampleBatches(datasetSize).size();
    }

    public TokenBudgetBatchReport report(int datasetSize) {
        return TokenBudgetBatchReport.from(lengths, sampleBatches(datasetSize), maxTokens, maxExamples);
    }

    public int[] lengths() {
        return Arrays.copyOf(lengths, lengths.length);
    }

    public int maxTokens() {
        return maxTokens;
    }

    public int maxExamples() {
        return maxExamples;
    }

    public boolean shuffleBatches() {
        return shuffleBatches;
    }

    public boolean shuffleWithinBatches() {
        return shuffleWithinBatches;
    }

    public long seed() {
        return seed;
    }

    private List<List<Integer>> buildSortedBatches() {
        List<Integer> sorted = sortedByLength();
        List<List<Integer>> batches = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        int currentMaxLength = 0;

        for (int index : sorted) {
            int length = lengths[index];
            int nextMaxLength = Math.max(currentMaxLength, length);
            if (!current.isEmpty() && wouldExceedBudget(nextMaxLength, current.size() + 1)) {
                batches.add(current);
                current = new ArrayList<>();
                currentMaxLength = 0;
                nextMaxLength = length;
            }

            current.add(index);
            currentMaxLength = nextMaxLength;
        }

        if (!current.isEmpty()) {
            batches.add(current);
        }
        return batches;
    }

    private boolean wouldExceedBudget(int maxLength, int exampleCount) {
        if (exampleCount > maxExamples) {
            return true;
        }
        return (long) maxLength * exampleCount > maxTokens;
    }

    private List<Integer> sortedByLength() {
        List<Integer> indices = new ArrayList<>(lengths.length);
        for (int i = 0; i < lengths.length; i++) {
            indices.add(i);
        }
        indices.sort(this::compareByLength);
        return indices;
    }

    private int compareByLength(int left, int right) {
        int comparison = Integer.compare(lengths[left], lengths[right]);
        return comparison != 0 ? comparison : Integer.compare(left, right);
    }

    private void requireDatasetSize(int datasetSize) {
        if (datasetSize != lengths.length) {
            throw new IllegalArgumentException(
                    "lengths must match dataset size, got " + lengths.length + " vs " + datasetSize);
        }
    }

    private static int[] validateLengths(int[] lengths) {
        Objects.requireNonNull(lengths, "lengths must not be null");
        int[] copy = Arrays.copyOf(lengths, lengths.length);
        for (int length : copy) {
            if (length < 0) {
                throw new IllegalArgumentException("lengths must be non-negative, got: " + length);
            }
        }
        return copy;
    }
}
