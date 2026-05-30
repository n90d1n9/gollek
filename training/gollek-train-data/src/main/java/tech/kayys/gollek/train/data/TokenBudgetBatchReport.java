package tech.kayys.gollek.train.data;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Diagnostics for one token-budget batch plan.
 */
public record TokenBudgetBatchReport(
        int sampleCount,
        int batchCount,
        int maxTokens,
        int maxExamples,
        int minBatchSize,
        int maxBatchSize,
        int maxObservedLength,
        int oversizedExampleCount,
        int[] batchSizes,
        int[] batchMaxLengths,
        long[] batchTokenCosts,
        long realTokens,
        long paddedTokens) {

    public TokenBudgetBatchReport {
        if (sampleCount < 0) {
            throw new IllegalArgumentException("sampleCount must be non-negative, got: " + sampleCount);
        }
        if (batchCount < 0) {
            throw new IllegalArgumentException("batchCount must be non-negative, got: " + batchCount);
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive, got: " + maxTokens);
        }
        if (maxExamples <= 0) {
            throw new IllegalArgumentException("maxExamples must be positive, got: " + maxExamples);
        }
        if (minBatchSize < 0 || maxBatchSize < 0 || minBatchSize > maxBatchSize) {
            throw new IllegalArgumentException("invalid batch size range");
        }
        if (maxObservedLength < 0) {
            throw new IllegalArgumentException("maxObservedLength must be non-negative, got: " + maxObservedLength);
        }
        if (oversizedExampleCount < 0) {
            throw new IllegalArgumentException(
                    "oversizedExampleCount must be non-negative, got: " + oversizedExampleCount);
        }
        batchSizes = Objects.requireNonNull(batchSizes, "batchSizes must not be null").clone();
        batchMaxLengths = Objects.requireNonNull(batchMaxLengths, "batchMaxLengths must not be null").clone();
        batchTokenCosts = Objects.requireNonNull(batchTokenCosts, "batchTokenCosts must not be null").clone();
        if (batchSizes.length != batchCount
                || batchMaxLengths.length != batchCount
                || batchTokenCosts.length != batchCount) {
            throw new IllegalArgumentException("batch detail arrays must match batchCount");
        }
        if (realTokens < 0 || paddedTokens < 0 || realTokens > paddedTokens) {
            throw new IllegalArgumentException("invalid real/padded token totals");
        }
    }

    static TokenBudgetBatchReport from(
            int[] lengths,
            List<List<Integer>> batches,
            int maxTokens,
            int maxExamples) {
        Objects.requireNonNull(lengths, "lengths must not be null");
        Objects.requireNonNull(batches, "batches must not be null");
        int[] batchSizes = new int[batches.size()];
        int[] batchMaxLengths = new int[batches.size()];
        long[] batchTokenCosts = new long[batches.size()];
        long realTokens = 0L;
        long paddedTokens = 0L;
        int minBatchSize = batches.isEmpty() ? 0 : Integer.MAX_VALUE;
        int maxBatchSize = 0;
        int maxObservedLength = 0;
        int oversizedExampleCount = 0;

        for (int i = 0; i < lengths.length; i++) {
            int length = lengths[i];
            if (length > maxTokens) {
                oversizedExampleCount++;
            }
            maxObservedLength = Math.max(maxObservedLength, length);
        }

        for (int i = 0; i < batches.size(); i++) {
            List<Integer> batch = Objects.requireNonNull(batches.get(i), "batch must not be null");
            int batchMaxLength = 0;
            long batchRealTokens = 0L;
            for (int index : batch) {
                if (index < 0 || index >= lengths.length) {
                    throw new IllegalArgumentException("batch index out of range: " + index);
                }
                int length = lengths[index];
                batchMaxLength = Math.max(batchMaxLength, length);
                batchRealTokens += length;
            }
            batchSizes[i] = batch.size();
            batchMaxLengths[i] = batchMaxLength;
            batchTokenCosts[i] = Math.multiplyExact((long) batchMaxLength, batch.size());
            minBatchSize = Math.min(minBatchSize, batch.size());
            maxBatchSize = Math.max(maxBatchSize, batch.size());
            realTokens = Math.addExact(realTokens, batchRealTokens);
            paddedTokens = Math.addExact(paddedTokens, batchTokenCosts[i]);
        }

        return new TokenBudgetBatchReport(
                lengths.length,
                batches.size(),
                maxTokens,
                maxExamples,
                minBatchSize,
                maxBatchSize,
                maxObservedLength,
                oversizedExampleCount,
                batchSizes,
                batchMaxLengths,
                batchTokenCosts,
                realTokens,
                paddedTokens);
    }

    @Override
    public int[] batchSizes() {
        return batchSizes.clone();
    }

    @Override
    public int[] batchMaxLengths() {
        return batchMaxLengths.clone();
    }

    @Override
    public long[] batchTokenCosts() {
        return batchTokenCosts.clone();
    }

    public long paddingTokens() {
        return paddedTokens - realTokens;
    }

    public double paddingRatio() {
        return paddedTokens == 0L ? 0.0 : (double) paddingTokens() / (double) paddedTokens;
    }

    public double utilization() {
        return paddedTokens == 0L ? 1.0 : (double) realTokens / (double) paddedTokens;
    }

    public double averageBatchSize() {
        return batchCount == 0 ? 0.0 : (double) sampleCount / (double) batchCount;
    }

    public double averageTokenCost() {
        return batchCount == 0 ? 0.0 : (double) paddedTokens / (double) batchCount;
    }

    public long maxBatchTokenCost() {
        return Arrays.stream(batchTokenCosts).max().orElse(0L);
    }
}
