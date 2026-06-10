package tech.kayys.gollek.ml.data;

import java.util.Objects;
import tech.kayys.gollek.train.data.internal.DataLoaderPaddingStatsRules;

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
        new DataLoaderPaddingStatsRules.Values(batchSize, maxLength, realTokens, paddedTokens);
    }

    public static PaddingStats fromLengths(int maxLength, int[] lengths) {
        DataLoaderPaddingStatsRules.Values values = DataLoaderPaddingStatsRules.fromLengths(maxLength, lengths);
        return fromValues(values);
    }

    public PaddingStats merge(PaddingStats other) {
        Objects.requireNonNull(other, "other must not be null");
        return fromValues(values().merge(other.values()));
    }

    public long paddingTokens() {
        return values().paddingTokens();
    }

    public double paddingRatio() {
        return values().paddingRatio();
    }

    public double utilization() {
        return values().utilization();
    }

    private DataLoaderPaddingStatsRules.Values values() {
        return new DataLoaderPaddingStatsRules.Values(batchSize, maxLength, realTokens, paddedTokens);
    }

    private static PaddingStats fromValues(DataLoaderPaddingStatsRules.Values values) {
        return new PaddingStats(values.batchSize(), values.maxLength(), values.realTokens(), values.paddedTokens());
    }
}
