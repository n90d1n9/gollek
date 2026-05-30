package tech.kayys.gollek.ml.data;

import java.util.Objects;

/**
 * Padding efficiency for one side of one or more padded sequence batches.
 *
 * <p>The counts are sequence positions, not tensor elements, so they stay stable
 * for token batches and feature-rich sequence tensors alike.</p>
 */
public record PaddingStats(
        int batchSize,
        int maxLength,
        long realTokens,
        long paddedTokens) {

    public PaddingStats {
        if (batchSize < 0) {
            throw new IllegalArgumentException("batchSize must be non-negative, got: " + batchSize);
        }
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be non-negative, got: " + maxLength);
        }
        if (realTokens < 0) {
            throw new IllegalArgumentException("realTokens must be non-negative, got: " + realTokens);
        }
        if (paddedTokens < 0) {
            throw new IllegalArgumentException("paddedTokens must be non-negative, got: " + paddedTokens);
        }
        if (realTokens > paddedTokens) {
            throw new IllegalArgumentException("realTokens cannot exceed paddedTokens");
        }
    }

    public static PaddingStats fromLengths(int maxLength, int[] lengths) {
        Objects.requireNonNull(lengths, "lengths must not be null");
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be non-negative, got: " + maxLength);
        }
        long realTokens = 0L;
        for (int length : lengths) {
            if (length < 0 || length > maxLength) {
                throw new IllegalArgumentException("lengths must be within the padded sequence length");
            }
            realTokens += length;
        }
        return new PaddingStats(
                lengths.length,
                maxLength,
                realTokens,
                Math.multiplyExact((long) lengths.length, maxLength));
    }

    public PaddingStats merge(PaddingStats other) {
        Objects.requireNonNull(other, "other must not be null");
        return new PaddingStats(
                Math.addExact(batchSize, other.batchSize),
                Math.max(maxLength, other.maxLength),
                Math.addExact(realTokens, other.realTokens),
                Math.addExact(paddedTokens, other.paddedTokens));
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
}
