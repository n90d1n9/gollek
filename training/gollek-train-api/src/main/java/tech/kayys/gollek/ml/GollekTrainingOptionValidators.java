package tech.kayys.gollek.ml;

/**
 * Validation and normalization helpers for {@link Gollek.DL.TrainingOptions}.
 */
final class GollekTrainingOptionValidators {
    private GollekTrainingOptionValidators() {
    }

    static Float normalizeFocalGamma(Float gamma) {
        if (gamma == null) {
            return null;
        }
        if (!Float.isFinite(gamma) || gamma < 0.0f) {
            throw new IllegalArgumentException("focalGamma must be finite and non-negative, got: " + gamma);
        }
        return gamma;
    }

    static String normalizeBestModelMonitorMetric(String metricName) {
        if (metricName == null || metricName.isBlank()) {
            return null;
        }
        String normalized = metricName.trim();
        if (normalized.startsWith("validationMetric.")) {
            normalized = normalized.substring("validationMetric.".length());
        }
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("bestModelMonitorMetric must not be blank");
        }
        return normalized;
    }

    static double normalizeNonNegativeDouble(double value) {
        return Double.isFinite(value) && value > 0.0 ? value : 0.0;
    }

    static Float normalizeFocalAlpha(Float alpha) {
        if (alpha == null) {
            return null;
        }
        if (!Float.isFinite(alpha) || alpha <= 0.0f) {
            throw new IllegalArgumentException("focalAlpha must be finite and positive, got: " + alpha);
        }
        return alpha;
    }

    static Float normalizeCausalLanguageModelingIgnoreIndex(Float ignoreIndex) {
        if (ignoreIndex == null) {
            return null;
        }
        if (!Float.isFinite(ignoreIndex)) {
            throw new IllegalArgumentException(
                    "causalLanguageModelingIgnoreIndex must be finite, got: " + ignoreIndex);
        }
        return ignoreIndex;
    }

    static float[] normalizeFocalClassWeights(float[] weights) {
        return normalizePositiveWeights(weights, "focalClassWeights");
    }

    static float[] normalizeCrossEntropyClassWeights(float[] weights) {
        return normalizePositiveWeights(weights, "crossEntropyClassWeights");
    }

    static float[] normalizeBcePositiveWeights(float[] weights) {
        return normalizePositiveWeights(weights, "bcePositiveWeights");
    }

    static float[] normalizePinballQuantiles(float[] quantiles) {
        if (quantiles == null) {
            return null;
        }
        if (quantiles.length == 0) {
            throw new IllegalArgumentException("pinballQuantiles must contain at least one value");
        }
        float[] copy = quantiles.clone();
        for (float quantile : copy) {
            if (!Float.isFinite(quantile) || quantile <= 0.0f || quantile >= 1.0f) {
                throw new IllegalArgumentException(
                        "pinballQuantiles must be finite values in (0, 1), got: " + quantile);
            }
        }
        return copy;
    }

    static Float normalizeIntervalAlpha(Float alpha) {
        if (alpha == null) {
            return null;
        }
        if (!Float.isFinite(alpha) || alpha <= 0.0f || alpha >= 1.0f) {
            throw new IllegalArgumentException("intervalAlpha must be finite and in (0, 1), got: " + alpha);
        }
        return alpha;
    }

    static Float normalizeIntervalCrossingPenalty(Float crossingPenalty) {
        if (crossingPenalty == null) {
            return null;
        }
        if (!Float.isFinite(crossingPenalty) || crossingPenalty <= 1.0f) {
            throw new IllegalArgumentException(
                    "intervalCrossingPenalty must be finite and greater than 1, got: " + crossingPenalty);
        }
        return crossingPenalty;
    }

    static Float normalizeTweediePower(Float power) {
        if (power == null) {
            return null;
        }
        if (!Float.isFinite(power) || power < 1.0f || power > 2.0f) {
            throw new IllegalArgumentException("tweediePower must be finite and in [1, 2], got: " + power);
        }
        return power;
    }

    static Float normalizePositiveFloat(Float value, String name) {
        if (value == null) {
            return null;
        }
        if (!Float.isFinite(value) || value <= 0.0f) {
            throw new IllegalArgumentException(name + " must be finite and positive, got: " + value);
        }
        return value;
    }

    static float[] toFloatArray(double[] values) {
        if (values == null) {
            return null;
        }
        float[] copy = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            copy[i] = (float) values[i];
        }
        return copy;
    }

    private static float[] normalizePositiveWeights(float[] weights, String name) {
        if (weights == null) {
            return null;
        }
        if (weights.length == 0) {
            throw new IllegalArgumentException(name + " must contain at least one value");
        }
        float[] copy = weights.clone();
        for (float weight : copy) {
            if (!Float.isFinite(weight) || weight <= 0.0f) {
                throw new IllegalArgumentException(name + " must be finite and positive, got: " + weight);
            }
        }
        return copy;
    }
}
