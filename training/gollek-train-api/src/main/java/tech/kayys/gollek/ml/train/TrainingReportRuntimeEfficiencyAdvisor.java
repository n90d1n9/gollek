package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Converts trainer runtime-efficiency summaries into focused optimization recommendations.
 */
final class TrainingReportRuntimeEfficiencyAdvisor {
    private static final double LOW_ACCOUNTED_PERCENT = 75.0;
    private static final double HIGH_OVERHEAD_PERCENT = 35.0;
    private static final double HIGH_BOTTLENECK_PERCENT = 75.0;

    private TrainingReportRuntimeEfficiencyAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(TrainingReportRuntimeEfficiency efficiency) {
        if (efficiency == null || !efficiency.available()) {
            return List.of();
        }
        List<TrainingReportRecommendation> recommendations = new ArrayList<>(3);
        if (efficiency.accountedPercent().isPresent()
                && efficiency.accountedPercent().orElseThrow() < LOW_ACCOUNTED_PERCENT) {
            recommendations.add(lowAccountedWallTime(efficiency));
        }
        if (efficiency.overheadPercent().isPresent()
                && efficiency.overheadPercent().orElseThrow() >= HIGH_OVERHEAD_PERCENT) {
            recommendations.add(wallClockOverhead(efficiency));
        }
        if (efficiency.bottleneckPercent().isPresent()
                && efficiency.bottleneckPercent().orElseThrow() >= HIGH_BOTTLENECK_PERCENT) {
            recommendations.add(dominantBottleneck(efficiency));
        }
        return List.copyOf(recommendations);
    }

    private static TrainingReportRecommendation lowAccountedWallTime(TrainingReportRuntimeEfficiency efficiency) {
        double accounted = efficiency.accountedPercent().orElseThrow();
        return new TrainingReportRecommendation(
                accounted < 50.0
                        ? TrainingReportRecommendation.Priority.HIGH
                        : TrainingReportRecommendation.Priority.MEDIUM,
                TrainingReportRecommendation.Category.REPORTING,
                accounted < 50.0
                        ? TrainingReportDiagnostics.Severity.WARNING
                        : TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.efficiency.low_accounted_wall_time",
                "Profile missing trainer wall-clock time",
                "The measured runtime-profile phases explain only part of the trainer wall-clock time.",
                List.of(
                        "Add scoped runtime profiling around unmeasured trainer loop work before tuning tensor kernels.",
                        "Compare measured phase total and wall-clock total after wrapping callbacks, logging, checkpointing, and data movement.",
                        "Keep this gate in CI for representative smoke runs so new unprofiled overhead is caught early."),
                evidence(efficiency));
    }

    private static TrainingReportRecommendation wallClockOverhead(TrainingReportRuntimeEfficiency efficiency) {
        String scope = efficiency.overheadScope();
        return new TrainingReportRecommendation(
                priority(efficiency.overheadPercent().orElseThrow(), 50.0),
                scopeCategory(scope),
                efficiency.overheadPercent().orElseThrow() >= 40.0
                        ? TrainingReportDiagnostics.Severity.WARNING
                        : TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.efficiency.wall_clock_overhead",
                "Reduce runtime overhead in `" + scope + "`",
                "The runtime-efficiency summary reports wall-clock scope time that is not explained by measured sub-phases.",
                overheadActions(scope),
                wallClockEvidence(efficiency));
    }

    private static TrainingReportRecommendation dominantBottleneck(TrainingReportRuntimeEfficiency efficiency) {
        String bottleneck = efficiency.bottleneck();
        return new TrainingReportRecommendation(
                priority(efficiency.bottleneckPercent().orElseThrow(), 85.0),
                bottleneckCategory(bottleneck),
                efficiency.bottleneckPercent().orElseThrow() >= 80.0
                        ? TrainingReportDiagnostics.Severity.WARNING
                        : TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.efficiency.dominant_bottleneck",
                "Optimize dominant trainer bottleneck `" + bottleneck + "`",
                "One trainer phase family dominates the measured runtime profile.",
                bottleneckActions(bottleneck),
                bottleneckEvidence(efficiency));
    }

    private static TrainingReportRecommendation.Priority priority(double value, double highThreshold) {
        return value >= highThreshold
                ? TrainingReportRecommendation.Priority.HIGH
                : TrainingReportRecommendation.Priority.MEDIUM;
    }

    private static TrainingReportRecommendation.Category scopeCategory(String scope) {
        return switch (scope) {
            case "optimizerStep" -> TrainingReportRecommendation.Category.OPTIMIZATION;
            case "validationBatch" -> TrainingReportRecommendation.Category.VALIDATION;
            case "trainBatch" -> TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
            default -> TrainingReportRecommendation.Category.REPORTING;
        };
    }

    private static TrainingReportRecommendation.Category bottleneckCategory(String bottleneck) {
        return switch (bottleneck) {
            case "input" -> TrainingReportRecommendation.Category.DATA_HEALTH;
            case "optimizer" -> TrainingReportRecommendation.Category.OPTIMIZATION;
            case "validation" -> TrainingReportRecommendation.Category.VALIDATION;
            case "compute", "train" -> TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
            default -> TrainingReportRecommendation.Category.REPORTING;
        };
    }

    private static List<String> overheadActions(String scope) {
        return switch (scope) {
            case "optimizerStep" -> List.of(
                    "Break down optimizer-step orchestration into diagnostics, clipping, scheduler, and state-update scopes.",
                    "Sample expensive optimizer diagnostics before changing model kernels.",
                    "Compare optimizer overhead before and after disabling optional debug scans.");
            case "validationBatch" -> List.of(
                    "Move validation-only metric setup and target conversion outside the hot batch path.",
                    "Reduce validation frequency or metric detail for short profiling runs.",
                    "Add focused profiling around validation callbacks and reporting hooks.");
            case "trainBatch" -> List.of(
                    "Inspect batch orchestration around callbacks, logging, guards, and metric hooks.",
                    "Move static per-batch setup out of the synchronous training loop.",
                    "Add a focused train-batch benchmark so framework overhead regressions fail early.");
            default -> List.of(
                    "Inspect trainer glue around the reported wall-clock scope.",
                    "Add profiling around callbacks, logging, checkpointing, and cross-device transfers.",
                    "Compare two short profiling runs before optimizing unrelated code.");
        };
    }

    private static List<String> bottleneckActions(String bottleneck) {
        return switch (bottleneck) {
            case "input" -> List.of(
                    "Enable or increase DataLoader prefetching when input time dominates measured runtime.",
                    "Move decoding, augmentation, dynamic padding, and target conversion out of the synchronous batch path when possible.",
                    "Compare train and validation input timings to locate the slow loader path.");
            case "optimizer" -> List.of(
                    "Profile optimizer diagnostics, clipping, scheduler, and parameter update phases separately.",
                    "Use sampled diagnostics for large models when full tensor scans dominate optimizer time.",
                    "Compare optimizer choices on a representative short run before changing the model.");
            case "validation" -> List.of(
                    "Review validation cadence, batch count, and metric cost before tuning training compute.",
                    "Cache validation-only preprocessing and metric state that does not change between batches.",
                    "Run a validation-only profile to separate forward, loss, and metric time.");
            case "compute", "train" -> List.of(
                    "Compare forward, backward, loss, and metric timings across batch sizes and backends.",
                    "Use the fastest available backend for the model path and verify device placement in the profile.",
                    "Add focused benchmarks for the dominant tensor operation before broad trainer changes.");
            default -> List.of(
                    "Inspect the dominant runtime group before tuning unrelated trainer settings.",
                    "Run a short repeat profile to confirm the bottleneck is stable.",
                    "Add a focused benchmark once the slow operation is identified.");
        };
    }

    private static Map<String, Object> evidence(TrainingReportRuntimeEfficiency efficiency) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("status", efficiency.status().name());
        efficiency.measuredMillis().ifPresent(value -> evidence.put("measuredMillis", value));
        efficiency.wallMillis().ifPresent(value -> evidence.put("wallMillis", value));
        efficiency.accountedPercent().ifPresent(value -> evidence.put("accountedPercent", value));
        evidence.put("overheadScope", efficiency.overheadScope());
        efficiency.overheadMillis().ifPresent(value -> evidence.put("overheadMillis", value));
        efficiency.overheadPercent().ifPresent(value -> evidence.put("overheadPercent", value));
        evidence.put("bottleneck", efficiency.bottleneck());
        efficiency.bottleneckPercent().ifPresent(value -> evidence.put("bottleneckPercent", value));
        evidence.put("primaryHotspot", efficiency.primaryHotspot());
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> wallClockEvidence(TrainingReportRuntimeEfficiency efficiency) {
        Map<String, Object> evidence = new LinkedHashMap<>(evidence(efficiency));
        evidence.put("scope", efficiency.overheadScope());
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> bottleneckEvidence(TrainingReportRuntimeEfficiency efficiency) {
        Map<String, Object> evidence = new LinkedHashMap<>(evidence(efficiency));
        evidence.put("group", efficiency.bottleneck());
        return Map.copyOf(evidence);
    }
}
