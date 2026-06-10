package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Interprets trainer wall-clock profiling data into reusable overhead diagnostics.
 */
final class TrainingReportRuntimeWallClockAssessment {
    private static final double DEFAULT_MIN_OVERHEAD_MILLIS = 1.0;
    private static final double DEFAULT_MIN_OVERHEAD_PERCENT = 15.0;
    static final Thresholds DEFAULT_THRESHOLDS =
            new Thresholds(DEFAULT_MIN_OVERHEAD_MILLIS, DEFAULT_MIN_OVERHEAD_PERCENT);

    private final boolean available;
    private final Scope scope;
    private final TrainingReportRuntimeProfile.WallScope overhead;
    private final Thresholds thresholds;

    private TrainingReportRuntimeWallClockAssessment(
            boolean available,
            Scope scope,
            TrainingReportRuntimeProfile.WallScope overhead,
            Thresholds thresholds) {
        this.available = available;
        this.scope = scope == null ? Scope.NONE : scope;
        this.overhead = overhead == null ? TrainingReportRuntimeProfile.WallScope.empty() : overhead;
        this.thresholds = thresholds == null ? DEFAULT_THRESHOLDS : thresholds;
    }

    static TrainingReportRuntimeWallClockAssessment assess(TrainingReportRuntimeProfile.WallClock wallClock) {
        return assess(wallClock, DEFAULT_THRESHOLDS);
    }

    static TrainingReportRuntimeWallClockAssessment assess(
            TrainingReportRuntimeProfile.WallClock wallClock,
            Thresholds thresholds) {
        if (wallClock == null || !wallClock.available()) {
            return new TrainingReportRuntimeWallClockAssessment(
                    false,
                    Scope.NONE,
                    TrainingReportRuntimeProfile.WallScope.empty(),
                    thresholds);
        }
        return new TrainingReportRuntimeWallClockAssessment(
                true,
                Scope.fromKey(wallClock.primaryOverheadScope()),
                wallClock.primaryOverhead(),
                thresholds);
    }

    boolean available() {
        return available;
    }

    Scope scope() {
        return scope;
    }

    String scopeKey() {
        return scope.key();
    }

    TrainingReportRuntimeProfile.WallScope overhead() {
        return overhead;
    }

    Thresholds thresholds() {
        return thresholds;
    }

    boolean overheadDetected() {
        return hasOverhead(thresholds.minOverheadMillis(), thresholds.minOverheadPercent());
    }

    boolean exceedsBudget(double maxOverheadPercent, double maxOverheadMillis) {
        double overheadPercent = overhead.overheadPercent().orElse(Double.NaN);
        double overheadMillis = overhead.overheadMillis().orElse(Double.NaN);
        boolean exceedsPercent = finite(overheadPercent) && finite(maxOverheadPercent)
                && overheadPercent > maxOverheadPercent;
        boolean exceedsMillis = finite(overheadMillis) && finite(maxOverheadMillis)
                && overheadMillis > maxOverheadMillis;
        return exceedsPercent || exceedsMillis;
    }

    TrainingReportRecommendation.Priority priority() {
        double percent = overhead.overheadPercent().orElse(0.0);
        if (percent >= 40.0) {
            return TrainingReportRecommendation.Priority.HIGH;
        }
        if (percent >= 25.0) {
            return TrainingReportRecommendation.Priority.MEDIUM;
        }
        return TrainingReportRecommendation.Priority.LOW;
    }

    TrainingReportDiagnostics.Severity severity() {
        return overhead.overheadPercent().orElse(0.0) >= 40.0
                ? TrainingReportDiagnostics.Severity.WARNING
                : TrainingReportDiagnostics.Severity.INFO;
    }

    TrainingReportRecommendation.Category category() {
        return switch (scope) {
            case OPTIMIZER_STEP -> TrainingReportRecommendation.Category.OPTIMIZATION;
            case VALIDATION_BATCH -> TrainingReportRecommendation.Category.VALIDATION;
            case TRAIN_BATCH, NONE -> TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
        };
    }

    Map<String, Object> recommendationEvidence(OptionalDouble wallTotalMillis, int scopeCount) {
        Map<String, Object> evidence = baseEvidence();
        put(evidence, "wallTotalMillis", wallTotalMillis);
        evidence.put("scopeCount", Math.max(0, scopeCount));
        return Map.copyOf(evidence);
    }

    Map<String, Object> budgetEvidence(
            OptionalDouble wallTotalMillis,
            double maxOverheadPercent,
            double maxOverheadMillis) {
        Map<String, Object> evidence = baseEvidence();
        putIfFinite(evidence, "thresholdPercent", maxOverheadPercent);
        putIfFinite(evidence, "thresholdMillis", maxOverheadMillis);
        put(evidence, "wallTotalMillis", wallTotalMillis);
        return Map.copyOf(evidence);
    }

    private Map<String, Object> baseEvidence() {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("scope", scopeKey());
        put(evidence, "totalMillis", overhead.totalMillis());
        put(evidence, "profiledMillis", overhead.profiledMillis());
        put(evidence, "overheadMillis", overhead.overheadMillis());
        put(evidence, "overheadPercent", overhead.overheadPercent());
        return evidence;
    }

    private boolean hasOverhead(double minMillis, double minPercent) {
        return overhead.overheadMillis().orElse(0.0) >= minMillis
                && overhead.overheadPercent().orElse(0.0) >= minPercent;
    }

    private static void put(Map<String, Object> evidence, String key, OptionalDouble value) {
        if (value != null) {
            value.ifPresent(number -> evidence.put(key, number));
        }
    }

    private static void putIfFinite(Map<String, Object> evidence, String key, double value) {
        if (finite(value)) {
            evidence.put(key, value);
        }
    }

    private static boolean finite(double value) {
        return Double.isFinite(value);
    }

    /**
     * Stable wall-clock trainer scope names exported by runtime-profile metadata.
     */
    enum Scope {
        TRAIN_BATCH("trainBatch"),
        VALIDATION_BATCH("validationBatch"),
        OPTIMIZER_STEP("optimizerStep"),
        NONE("none");

        private final String key;

        Scope(String key) {
            this.key = key;
        }

        String key() {
            return key;
        }

        static Scope fromKey(String key) {
            return switch (key == null ? "" : key.trim()) {
                case "trainBatch" -> TRAIN_BATCH;
                case "validationBatch" -> VALIDATION_BATCH;
                case "optimizerStep" -> OPTIMIZER_STEP;
                default -> NONE;
            };
        }
    }

    /**
     * Minimum wall-clock overhead needed before runtime advice is emitted.
     */
    record Thresholds(double minOverheadMillis, double minOverheadPercent) {
        Thresholds {
            minOverheadMillis = positiveOrDefault(minOverheadMillis, DEFAULT_MIN_OVERHEAD_MILLIS);
            minOverheadPercent = positiveOrDefault(minOverheadPercent, DEFAULT_MIN_OVERHEAD_PERCENT);
        }

        private static double positiveOrDefault(double value, double fallback) {
            return Double.isFinite(value) && value > 0.0 ? value : fallback;
        }
    }
}
