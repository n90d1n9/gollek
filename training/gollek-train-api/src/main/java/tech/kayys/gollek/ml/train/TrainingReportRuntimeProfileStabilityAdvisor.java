package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Detects runtime phases whose latency is spiky enough to deserve focused profiling.
 */
final class TrainingReportRuntimeProfileStabilityAdvisor {
    private static final long MIN_SAMPLES = 2L;
    private static final double MIN_COEFFICIENT_OF_VARIATION = 0.35;
    private static final double MIN_MAX_TO_AVERAGE_RATIO = 1.40;

    private TrainingReportRuntimeProfileStabilityAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(TrainingReportRuntimeProfile profile) {
        if (profile == null || !profile.available()) {
            return List.of();
        }
        Optional<TrainingReportRuntimeProfile.Group> primaryGroup = primaryGroupWithDetails(profile);
        if (primaryGroup.isEmpty()) {
            return List.of();
        }
        List<TrainingReportRecommendation> recommendations = new java.util.ArrayList<>(2);
        unstablePrimaryGroup(profile, primaryGroup.get()).ifPresent(recommendations::add);
        spikyPrimaryGroup(profile, primaryGroup.get()).ifPresent(recommendations::add);
        return List.copyOf(recommendations);
    }

    private static Optional<TrainingReportRecommendation> unstablePrimaryGroup(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group) {
        if (group.name().isBlank()
                || group.count().orElse(0L) < MIN_SAMPLES
                || group.averageMillis().isEmpty()
                || group.stddevMillis().isEmpty()) {
            return Optional.empty();
        }
        double averageMillis = group.averageMillis().orElseThrow();
        double stddevMillis = group.stddevMillis().orElseThrow();
        if (averageMillis <= 0.0) {
            return Optional.empty();
        }
        double coefficientOfVariation = stddevMillis / averageMillis;
        if (coefficientOfVariation < MIN_COEFFICIENT_OF_VARIATION) {
            return Optional.empty();
        }
        return Optional.of(new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                category(group.name()),
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.primary_group_variability",
                "Stabilize trainer runtime group `" + group.name() + "`",
                "The primary runtime group has high latency variation, so averages alone may hide intermittent stalls.",
                actions(group.name()),
                evidence(profile, group, coefficientOfVariation)));
    }

    private static Optional<TrainingReportRecommendation> spikyPrimaryGroup(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group) {
        if (group.name().isBlank()
                || group.count().orElse(0L) < MIN_SAMPLES
                || group.averageMillis().isEmpty()
                || group.maxMillis().isEmpty()) {
            return Optional.empty();
        }
        double averageMillis = group.averageMillis().orElseThrow();
        double maxMillis = group.maxMillis().orElseThrow();
        if (averageMillis <= 0.0) {
            return Optional.empty();
        }
        double maxToAverageRatio = maxMillis / averageMillis;
        if (maxToAverageRatio < MIN_MAX_TO_AVERAGE_RATIO) {
            return Optional.empty();
        }
        return Optional.of(new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                category(group.name()),
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.primary_group_spike",
                "Investigate worst-case trainer runtime spike in `" + group.name() + "`",
                "The primary runtime group's slowest sample is much higher than its average, which can hurt tail latency and throughput predictability.",
                spikeActions(group.name()),
                spikeEvidence(profile, group, maxToAverageRatio)));
    }

    private static Optional<TrainingReportRuntimeProfile.Group> primaryGroupWithDetails(
            TrainingReportRuntimeProfile profile) {
        Optional<TrainingReportRuntimeProfile.Group> primary = profile.primaryGroup();
        if (primary.isEmpty()) {
            return Optional.empty();
        }
        String primaryName = primary.get().name();
        return profile.groups().stream()
                .filter(group -> group.name().equals(primaryName))
                .findFirst()
                .or(() -> primary);
    }

    private static TrainingReportRecommendation.Category category(String group) {
        if ("optimizer".equals(group)) {
            return TrainingReportRecommendation.Category.OPTIMIZATION;
        }
        if ("input".equals(group)) {
            return TrainingReportRecommendation.Category.DATA_HEALTH;
        }
        return TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
    }

    private static List<String> actions(String group) {
        return switch (group) {
            case "optimizer" -> List.of(
                    "Profile optimizer diagnostics separately from the optimizer step to find scan-heavy spikes.",
                    "Check whether gradient clipping, parameter validation, or scheduler work runs intermittently.",
                    "Compare variability before and after sampling expensive optimizer diagnostics.");
            case "input" -> List.of(
                    "Inspect input iterator, hasNext, and next timing separately to locate loader jitter.",
                    "Check whether decoding, augmentation, shuffling, or remote/disk reads occasionally block the training loop.",
                    "Increase prefetch buffering or cache deterministic transforms when input variability stays high.");
            case "validation" -> List.of(
                    "Check validation batch sizes and preprocessing paths for uneven work.",
                    "Separate validation forward, loss, and metric timing before changing training settings.",
                    "Run validation with a fixed small sample window to confirm whether the spike is data-dependent.");
            case "train" -> List.of(
                    "Break the train group down by forward, backward, loss, metrics, and batch adaptation timing.",
                    "Inspect data-loader and preprocessing jitter before tuning model math.",
                    "Compare the coefficient of variation across CPU and GPU/Metal backends to isolate backend stalls.");
            default -> List.of(
                    "Inspect the phase-level runtime table to find which child phase is spiky.",
                    "Run two short profiles with the same seed and data order to separate noise from regression.",
                    "Add a focused microbenchmark once the unstable phase is identified.");
        };
    }

    private static List<String> spikeActions(String group) {
        return switch (group) {
            case "optimizer" -> List.of(
                    "Inspect the slowest optimizer sample and split diagnostics, clipping, scheduler, and parameter update timing.",
                    "Check whether occasional full tensor scans are still enabled inside optimizer-side diagnostics.",
                    "Use sampled diagnostics or smaller diagnostic windows when the max-to-average ratio stays high.");
            case "input" -> List.of(
                    "Inspect the slowest input sample before changing model or optimizer settings.",
                    "Check whether the spike comes from batch construction, decoding, augmentation, dynamic padding, or storage latency.",
                    "Use prefetching or cached transforms when input max-to-average ratio remains high.");
            case "validation" -> List.of(
                    "Identify whether the max validation sample comes from a specific batch, metric, or preprocessing path.",
                    "Use fixed-size validation batches when comparing trainer runtime changes.",
                    "Cache validation-only transforms that create occasional long samples.");
            case "train" -> List.of(
                    "Inspect the worst train sample before optimizing the average path.",
                    "Check data loading, dynamic padding, target conversion, and backend transfer boundaries around the spike.",
                    "Record a short profile with per-child train phases to separate data jitter from model compute.");
            default -> List.of(
                    "Inspect the slowest sample before optimizing the average runtime.",
                    "Compare max-to-average ratio across two deterministic runs.",
                    "Add a focused benchmark around the phase that owns the spike.");
        };
    }

    private static Map<String, Object> evidence(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group,
            double coefficientOfVariation) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("group", group.name());
        evidence.put("groupCount", profile.groupCount());
        evidence.put("coefficientOfVariation", coefficientOfVariation);
        evidence.put("threshold", MIN_COEFFICIENT_OF_VARIATION);
        group.count().ifPresent(value -> evidence.put("count", value));
        group.averageMillis().ifPresent(value -> evidence.put("averageMillis", value));
        group.stddevMillis().ifPresent(value -> evidence.put("stddevMillis", value));
        group.minMillis().ifPresent(value -> evidence.put("minMillis", value));
        group.maxMillis().ifPresent(value -> evidence.put("maxMillis", value));
        group.lastMillis().ifPresent(value -> evidence.put("lastMillis", value));
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> spikeEvidence(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group,
            double maxToAverageRatio) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("group", group.name());
        evidence.put("groupCount", profile.groupCount());
        evidence.put("maxToAverageRatio", maxToAverageRatio);
        evidence.put("threshold", MIN_MAX_TO_AVERAGE_RATIO);
        group.count().ifPresent(value -> evidence.put("count", value));
        group.averageMillis().ifPresent(value -> evidence.put("averageMillis", value));
        group.maxMillis().ifPresent(value -> evidence.put("maxMillis", value));
        group.minMillis().ifPresent(value -> evidence.put("minMillis", value));
        group.lastMillis().ifPresent(value -> evidence.put("lastMillis", value));
        group.stddevMillis().ifPresent(value -> evidence.put("stddevMillis", value));
        return Map.copyOf(evidence);
    }
}
