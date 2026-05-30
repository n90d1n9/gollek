package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;

/**
 * Backend-neutral weighting config for GRAM-style variational recursive training.
 */
public record GramObjectiveConfig(
        double reconstructionWeight,
        double klBeta,
        double klBalance,
        double latentProcessRewardWeight,
        double adaptiveComputationWeight,
        GramObjectiveReduction reduction,
        boolean truncatedSurrogate) {

    public GramObjectiveConfig {
        reconstructionWeight = requireNonNegativeFinite(reconstructionWeight, "reconstructionWeight");
        klBeta = requireNonNegativeFinite(klBeta, "klBeta");
        klBalance = requireRange(klBalance, "klBalance", 0.0, 1.0);
        latentProcessRewardWeight = requireNonNegativeFinite(
                latentProcessRewardWeight,
                "latentProcessRewardWeight");
        adaptiveComputationWeight = requireNonNegativeFinite(
                adaptiveComputationWeight,
                "adaptiveComputationWeight");
        reduction = Objects.requireNonNullElse(reduction, GramObjectiveReduction.MEAN);
    }

    public double effectiveKlWeight() {
        return klBeta;
    }

    public static Builder builder() {
        return new Builder();
    }

    private static double requireNonNegativeFinite(double value, String name) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(name + " must be finite and >= 0 but was " + value);
        }
        return value;
    }

    private static double requireRange(double value, String name, double minInclusive, double maxInclusive) {
        if (!Double.isFinite(value) || value < minInclusive || value > maxInclusive) {
            throw new IllegalArgumentException(
                    name + " must be finite and in [" + minInclusive + ", " + maxInclusive + "] but was " + value);
        }
        return value;
    }

    public static final class Builder {
        private double reconstructionWeight = 1.0;
        private double klBeta = 0.1;
        private double klBalance = 1.0;
        private double latentProcessRewardWeight;
        private double adaptiveComputationWeight;
        private GramObjectiveReduction reduction = GramObjectiveReduction.MEAN;
        private boolean truncatedSurrogate = true;

        private Builder() {
        }

        public Builder reconstructionWeight(double reconstructionWeight) {
            this.reconstructionWeight = reconstructionWeight;
            return this;
        }

        public Builder klBeta(double klBeta) {
            this.klBeta = klBeta;
            return this;
        }

        public Builder klBalance(double klBalance) {
            this.klBalance = klBalance;
            return this;
        }

        public Builder latentProcessRewardWeight(double latentProcessRewardWeight) {
            this.latentProcessRewardWeight = latentProcessRewardWeight;
            return this;
        }

        public Builder adaptiveComputationWeight(double adaptiveComputationWeight) {
            this.adaptiveComputationWeight = adaptiveComputationWeight;
            return this;
        }

        public Builder reduction(GramObjectiveReduction reduction) {
            this.reduction = reduction;
            return this;
        }

        public Builder truncatedSurrogate(boolean truncatedSurrogate) {
            this.truncatedSurrogate = truncatedSurrogate;
            return this;
        }

        public GramObjectiveConfig build() {
            return new GramObjectiveConfig(
                    reconstructionWeight,
                    klBeta,
                    klBalance,
                    latentProcessRewardWeight,
                    adaptiveComputationWeight,
                    reduction,
                    truncatedSurrogate);
        }
    }
}
