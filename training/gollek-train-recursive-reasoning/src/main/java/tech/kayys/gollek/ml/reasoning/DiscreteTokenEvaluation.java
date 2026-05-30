package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Benchmark-neutral evaluation of one discrete token candidate.
 */
public record DiscreteTokenEvaluation(
        boolean valid,
        String canonicalKey,
        int errorCount,
        Map<String, Object> metadata) {

    public DiscreteTokenEvaluation {
        if (valid && (canonicalKey == null || canonicalKey.isBlank())) {
            throw new IllegalArgumentException("canonicalKey must be present for valid candidates");
        }
        if (canonicalKey != null && canonicalKey.isBlank()) {
            throw new IllegalArgumentException("canonicalKey must not be blank");
        }
        if (errorCount < 0) {
            throw new IllegalArgumentException("errorCount must be >= 0 but was " + errorCount);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public static DiscreteTokenEvaluation valid(String canonicalKey) {
        return valid(canonicalKey, Map.of());
    }

    public static DiscreteTokenEvaluation valid(String canonicalKey, Map<String, Object> metadata) {
        return new DiscreteTokenEvaluation(true, canonicalKey, 0, metadata);
    }

    public static DiscreteTokenEvaluation invalid(int errorCount) {
        return invalid(errorCount, Map.of());
    }

    public static DiscreteTokenEvaluation invalid(int errorCount, Map<String, Object> metadata) {
        return new DiscreteTokenEvaluation(false, null, errorCount, metadata);
    }
}
