package tech.kayys.gollek.ml.data;

import java.util.Objects;

/**
 * Aggregate padding efficiency for a padded input/label batch.
 */
public record PaddingEfficiencyReport(
        PaddingStats inputs,
        PaddingStats labels) {

    public PaddingEfficiencyReport {
        inputs = Objects.requireNonNull(inputs, "inputs must not be null");
        labels = Objects.requireNonNull(labels, "labels must not be null");
    }

    public long realTokens() {
        return Math.addExact(inputs.realTokens(), labels.realTokens());
    }

    public long paddedTokens() {
        return Math.addExact(inputs.paddedTokens(), labels.paddedTokens());
    }

    public long paddingTokens() {
        return Math.addExact(inputs.paddingTokens(), labels.paddingTokens());
    }

    public double paddingRatio() {
        long padded = paddedTokens();
        return padded == 0L ? 0.0 : (double) paddingTokens() / (double) padded;
    }

    public double utilization() {
        long padded = paddedTokens();
        return padded == 0L ? 1.0 : (double) realTokens() / (double) padded;
    }
}
