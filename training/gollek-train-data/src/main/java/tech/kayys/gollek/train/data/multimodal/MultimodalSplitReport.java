package tech.kayys.gollek.train.data.multimodal;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Data-quality report for an already materialized train/validation split.
 */
public record MultimodalSplitReport(
        MultimodalDatasetReport trainReport,
        MultimodalDatasetReport validationReport,
        Set<String> overlappingSourcePaths,
        Map<String, Double> sampleSignatureShareDelta,
        Map<String, Double> mimeTypeShareDelta) {
    public MultimodalSplitReport {
        trainReport = Objects.requireNonNull(trainReport, "trainReport must not be null");
        validationReport = Objects.requireNonNull(validationReport, "validationReport must not be null");
        overlappingSourcePaths = immutableStringSet(overlappingSourcePaths, "overlappingSourcePaths");
        sampleSignatureShareDelta = immutableDeltaMap(sampleSignatureShareDelta, "sampleSignatureShareDelta");
        mimeTypeShareDelta = immutableDeltaMap(mimeTypeShareDelta, "mimeTypeShareDelta");
    }

    public int totalSampleCount() {
        return trainReport.sampleCount() + validationReport.sampleCount();
    }

    public double trainSampleFraction() {
        return sampleFraction(trainReport.sampleCount());
    }

    public double validationSampleFraction() {
        return sampleFraction(validationReport.sampleCount());
    }

    public boolean hasSourceLeakage() {
        return !overlappingSourcePaths.isEmpty();
    }

    public double maxSampleSignatureShareDelta() {
        return maxDelta(sampleSignatureShareDelta);
    }

    public double maxMimeTypeShareDelta() {
        return maxDelta(mimeTypeShareDelta);
    }

    public boolean isSignatureBalanced(double tolerance) {
        requireTolerance(tolerance);
        return maxSampleSignatureShareDelta() <= tolerance;
    }

    public boolean isMimeTypeBalanced(double tolerance) {
        requireTolerance(tolerance);
        return maxMimeTypeShareDelta() <= tolerance;
    }

    public void throwIfSourceLeakage() {
        if (hasSourceLeakage()) {
            throw new IllegalStateException("multimodal split leaks source assets across train/validation: "
                    + overlappingSourcePaths);
        }
    }

    public void throwIfInvalid(double signatureTolerance, double mimeTypeTolerance) {
        requireTolerance(signatureTolerance);
        requireTolerance(mimeTypeTolerance);
        Map<String, Object> failures = new LinkedHashMap<>();
        if (hasSourceLeakage()) {
            failures.put("sourceLeakage", overlappingSourcePaths);
        }
        if (!isSignatureBalanced(signatureTolerance)) {
            failures.put("signatureDrift", maxSampleSignatureShareDelta());
        }
        if (!isMimeTypeBalanced(mimeTypeTolerance)) {
            failures.put("mimeTypeDrift", maxMimeTypeShareDelta());
        }
        if (!failures.isEmpty()) {
            throw new IllegalStateException("multimodal split audit failed: " + failures);
        }
    }

    private double sampleFraction(int count) {
        int total = totalSampleCount();
        return total == 0 ? 0.0 : (double) count / total;
    }

    private static double maxDelta(Map<String, Double> deltas) {
        double max = 0.0;
        for (double delta : deltas.values()) {
            max = Math.max(max, delta);
        }
        return max;
    }

    static void requireTolerance(double tolerance) {
        if (!Double.isFinite(tolerance) || tolerance < 0.0) {
            throw new IllegalArgumentException("tolerance must be finite and non-negative");
        }
    }

    private static Set<String> immutableStringSet(Set<String> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        TreeSet<String> copy = new TreeSet<>();
        for (String value : values) {
            String normalized = Objects.requireNonNull(value, name + " values must not be null").trim();
            if (normalized.isEmpty()) {
                throw new IllegalArgumentException(name + " values must not be blank");
            }
            copy.add(normalized);
        }
        return Collections.unmodifiableSet(copy);
    }

    private static Map<String, Double> immutableDeltaMap(Map<String, Double> values, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        Map<String, Double> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            String normalized = Objects.requireNonNull(key, name + " key must not be null").trim();
            if (normalized.isEmpty()) {
                throw new IllegalArgumentException(name + " keys must not be blank");
            }
            if (value == null || !Double.isFinite(value) || value < 0.0 || value > 1.0) {
                throw new IllegalArgumentException(name + " values must be finite and between 0 and 1");
            }
            copy.put(normalized, value);
        });
        return Collections.unmodifiableMap(copy);
    }
}
