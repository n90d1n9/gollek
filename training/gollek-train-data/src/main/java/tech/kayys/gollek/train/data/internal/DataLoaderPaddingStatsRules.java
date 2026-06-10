package tech.kayys.gollek.train.data.internal;

import java.util.Objects;

public final class DataLoaderPaddingStatsRules {
    private DataLoaderPaddingStatsRules() {
    }

    public record Values(
            int batchSize,
            int maxLength,
            long realTokens,
            long paddedTokens) {

        public Values {
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

        public Values merge(Values other) {
            Objects.requireNonNull(other, "other must not be null");
            return new Values(
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

    public static Values fromLengths(int maxLength, int[] lengths) {
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
        return new Values(
                lengths.length,
                maxLength,
                realTokens,
                Math.multiplyExact((long) lengths.length, maxLength));
    }
}
