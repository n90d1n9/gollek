package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Classifies aggregate trainer runtime-balance data against reusable performance thresholds.
 */
public record TrainingReportRuntimeProfileBalanceAssessment(
        boolean available,
        TrainingReportRuntimeProfile.BalanceBucket dominantBucket,
        OptionalDouble dominantPercent,
        boolean inputBound,
        boolean optimizerBound,
        boolean validationBound,
        boolean trainComputeBound,
        Thresholds thresholds,
        TrainingReportRuntimeProfile.Balance balance) {
    public TrainingReportRuntimeProfileBalanceAssessment {
        dominantBucket = dominantBucket == null
                ? TrainingReportRuntimeProfile.BalanceBucket.NONE
                : dominantBucket;
        dominantPercent = dominantPercent == null ? OptionalDouble.empty() : dominantPercent;
        thresholds = thresholds == null ? Thresholds.advisoryDefaults() : thresholds;
        balance = balance == null ? TrainingReportRuntimeProfile.Balance.empty() : balance;
        available = available || balance.available();
    }

    public static TrainingReportRuntimeProfileBalanceAssessment assess(
            TrainingReportRuntimeProfile.Balance balance) {
        return assess(balance, Thresholds.advisoryDefaults());
    }

    public static TrainingReportRuntimeProfileBalanceAssessment assess(
            TrainingReportRuntimeProfile.Balance balance,
            Thresholds thresholds) {
        TrainingReportRuntimeProfile.Balance resolvedBalance =
                balance == null ? TrainingReportRuntimeProfile.Balance.empty() : balance;
        Thresholds resolvedThresholds = thresholds == null ? Thresholds.advisoryDefaults() : thresholds;
        return new TrainingReportRuntimeProfileBalanceAssessment(
                resolvedBalance.available(),
                resolvedBalance.dominantBucket(),
                resolvedBalance.dominantPercent(),
                meets(resolvedBalance.input(), resolvedThresholds.inputBoundPercent()),
                meets(resolvedBalance.optimizer(), resolvedThresholds.optimizerBoundPercent()),
                meets(resolvedBalance.validation(), resolvedThresholds.validationBoundPercent()),
                meets(resolvedBalance.train(), resolvedThresholds.trainComputeBoundPercent()),
                resolvedThresholds,
                resolvedBalance);
    }

    public boolean requiresAttention() {
        return inputBound || optimizerBound || validationBound || trainComputeBound;
    }

    public OptionalDouble totalMillis(TrainingReportRuntimeProfile.BalanceBucket bucket) {
        return balance.bucket(bucket).totalMillis();
    }

    public OptionalDouble percentTotal(TrainingReportRuntimeProfile.BalanceBucket bucket) {
        return balance.bucket(bucket).percentTotal();
    }

    public boolean exceeds(TrainingReportRuntimeProfile.BalanceBucket bucket, double thresholdPercent) {
        return Double.isFinite(thresholdPercent)
                && percentTotal(bucket).isPresent()
                && percentTotal(bucket).orElseThrow() > thresholdPercent;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available);
        map.put("requiresAttention", requiresAttention());
        map.put("dominantBucket", dominantBucket.name());
        dominantPercent.ifPresent(value -> map.put("dominantPercent", value));
        map.put("inputBound", inputBound);
        map.put("optimizerBound", optimizerBound);
        map.put("validationBound", validationBound);
        map.put("trainComputeBound", trainComputeBound);
        map.put("thresholds", thresholds.toMap());
        map.put("balance", balance.toMap());
        return Map.copyOf(map);
    }

    Map<String, Object> recommendationEvidence() {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("bottleneckGroup", balance.bottleneckGroup());
        evidence.put("dominantBucket", dominantBucket.name());
        dominantPercent.ifPresent(value -> evidence.put("dominantPercent", value));
        evidence.put("totalMillis", balance.totalMillis().orElse(0.0));
        appendBucket(evidence, "input", balance.input());
        appendBucket(evidence, "compute", balance.compute());
        appendBucket(evidence, "train", balance.train());
        appendBucket(evidence, "validation", balance.validation());
        appendBucket(evidence, "optimizer", balance.optimizer());
        return Map.copyOf(evidence);
    }

    private static void appendBucket(
            Map<String, Object> evidence,
            String prefix,
            TrainingReportRuntimeProfile.Bucket bucket) {
        evidence.put(prefix + "Millis", bucket.totalMillis().orElse(0.0));
        evidence.put(prefix + "Percent", bucket.percentTotal().orElse(0.0));
    }

    private static boolean meets(TrainingReportRuntimeProfile.Bucket bucket, double thresholdPercent) {
        return Double.isFinite(thresholdPercent)
                && bucket != null
                && bucket.percentTotal().isPresent()
                && bucket.percentTotal().orElseThrow() >= thresholdPercent;
    }

    /**
     * Stable thresholds used to classify runtime-balance bottlenecks.
     */
    public record Thresholds(
            double inputBoundPercent,
            double optimizerBoundPercent,
            double validationBoundPercent,
            double trainComputeBoundPercent) {
        public static Thresholds advisoryDefaults() {
            return new Thresholds(50.0, 45.0, 50.0, 70.0);
        }

        public Map<String, Object> toMap() {
            return Map.of(
                    "inputBoundPercent", inputBoundPercent,
                    "optimizerBoundPercent", optimizerBoundPercent,
                    "validationBoundPercent", validationBoundPercent,
                    "trainComputeBoundPercent", trainComputeBoundPercent);
        }
    }
}
