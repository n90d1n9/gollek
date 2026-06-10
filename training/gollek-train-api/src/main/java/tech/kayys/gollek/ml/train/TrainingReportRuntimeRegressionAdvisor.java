package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Compares candidate trainer runtime profile timings against a baseline report.
 */
final class TrainingReportRuntimeRegressionAdvisor {
    private static final double MIN_GROUP_AVERAGE_REGRESSION_RATIO = 1.15;
    private static final double MIN_HOTSPOT_AVERAGE_REGRESSION_RATIO = 1.20;
    private static final double MIN_ACCOUNTED_WALL_TIME_DROP_PERCENT = 15.0;
    private static final double MIN_WALL_CLOCK_OVERHEAD_INCREASE_PERCENT = 10.0;
    private static final double MIN_BOTTLENECK_INCREASE_PERCENT = 15.0;

    private TrainingReportRuntimeRegressionAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(
            Map<String, ?> baselineReport,
            Map<String, ?> candidateReport) {
        TrainingReportRuntimeProfile baselineProfile = TrainingReportReader.runtimeProfileView(baselineReport);
        TrainingReportRuntimeProfile candidateProfile = TrainingReportReader.runtimeProfileView(candidateReport);
        TrainingReportRuntimeRegressionSummary summary = summary(baselineProfile, candidateProfile);
        java.util.ArrayList<TrainingReportRecommendation> recommendations = new java.util.ArrayList<>(2);
        groupAverageRegression(baselineProfile, candidateProfile, summary).ifPresent(recommendations::add);
        hotspotAverageRegression(baselineProfile, candidateProfile, summary).ifPresent(recommendations::add);
        accountedWallTimeRegression(summary).ifPresent(recommendations::add);
        wallClockOverheadRegression(summary).ifPresent(recommendations::add);
        dominantBottleneckRegression(summary).ifPresent(recommendations::add);
        return List.copyOf(recommendations);
    }

    static TrainingReportRuntimeRegressionSummary summary(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate) {
        if (baseline == null || candidate == null || !baseline.available() || !candidate.available()) {
            return TrainingReportRuntimeRegressionSummary.empty();
        }
        TrainingReportRuntimeEfficiency baselineEfficiency = TrainingReportRuntimeEfficiency.from(baseline);
        TrainingReportRuntimeEfficiency candidateEfficiency = TrainingReportRuntimeEfficiency.from(candidate);
        return new TrainingReportRuntimeRegressionSummary(
                groupAverageEntry(baseline, candidate),
                hotspotAverageEntry(baseline, candidate),
                accountedWallTimeEntry(baselineEfficiency, candidateEfficiency),
                wallClockOverheadEntry(baselineEfficiency, candidateEfficiency),
                dominantBottleneckEntry(baselineEfficiency, candidateEfficiency));
    }

    private static Optional<TrainingReportRecommendation> accountedWallTimeRegression(
            TrainingReportRuntimeRegressionSummary summary) {
        return summary.accountedWallTime()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .map(entry -> new TrainingReportRecommendation(
                        TrainingReportRecommendation.Priority.HIGH,
                        TrainingReportRecommendation.Category.REPORTING,
                        TrainingReportDiagnostics.Severity.WARNING,
                        "runtime_profile.efficiency.accounted_wall_time_regressed",
                        "Restore runtime profile wall-clock coverage",
                        "The candidate explains less trainer wall-clock time with measured runtime phases than the baseline.",
                        List.of(
                                "Compare baseline and candidate profiler scope coverage before tuning tensor kernels.",
                                "Add scoped profiling around new callbacks, logging, checkpointing, data movement, and training-loop glue.",
                                "Keep the baseline report available until the unprofiled candidate wall time is isolated."),
                        efficiencyEvidence(entry)));
    }

    private static Optional<TrainingReportRecommendation> wallClockOverheadRegression(
            TrainingReportRuntimeRegressionSummary summary) {
        return summary.wallClockOverhead()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .map(entry -> new TrainingReportRecommendation(
                        TrainingReportRecommendation.Priority.HIGH,
                        TrainingReportRecommendation.Category.TRAINING_DYNAMICS,
                        TrainingReportDiagnostics.Severity.WARNING,
                        "runtime_profile.efficiency.wall_clock_overhead_regressed",
                        "Reduce regressed wall-clock overhead in `" + entry.key() + "`",
                        "The candidate runtime profile has more unexplained wall-clock overhead than the baseline.",
                        List.of(
                                "Compare candidate overhead scope timing against the baseline before changing model math.",
                                "Inspect callbacks, logging, metric hooks, checkpointing, and cross-device transfers added around the scope.",
                                "Add a focused overhead regression benchmark for this scope once reproduced."),
                        efficiencyEvidence(entry)));
    }

    private static Optional<TrainingReportRecommendation> dominantBottleneckRegression(
            TrainingReportRuntimeRegressionSummary summary) {
        return summary.dominantBottleneck()
                .filter(TrainingReportRuntimeRegressionSummary.EfficiencyEntry::regressed)
                .map(entry -> new TrainingReportRecommendation(
                        TrainingReportRecommendation.Priority.HIGH,
                        category(entry.key()),
                        TrainingReportDiagnostics.Severity.WARNING,
                        "runtime_profile.efficiency.dominant_bottleneck_regressed",
                        "Reduce regressed runtime bottleneck `" + entry.key() + "`",
                        "The candidate spends a larger share of measured runtime in the dominant bottleneck than the baseline.",
                        bottleneckRegressionActions(entry.key()),
                        efficiencyEvidence(entry)));
    }

    private static Optional<TrainingReportRecommendation> groupAverageRegression(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate,
            TrainingReportRuntimeRegressionSummary summary) {
        Optional<TrainingReportRuntimeRegressionSummary.Entry> maybeEntry = summary.primaryGroupAverage()
                .filter(TrainingReportRuntimeRegressionSummary.Entry::regressed);
        if (maybeEntry.isEmpty()) {
            return Optional.empty();
        }
        Optional<TrainingReportRuntimeProfile.Group> maybeCandidate = candidate.primaryGroup();
        if (maybeCandidate.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Group candidateGroup = detailedGroup(candidate, maybeCandidate.get());
        Optional<TrainingReportRuntimeProfile.Group> maybeBaseline = baseline.groups().stream()
                .filter(group -> group.name().equals(candidateGroup.name()))
                .findFirst();
        if (maybeBaseline.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Group baselineGroup = maybeBaseline.get();
        TrainingReportRuntimeRegressionSummary.Entry regression = maybeEntry.get();
        return Optional.of(new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.HIGH,
                category(candidateGroup.name()),
                TrainingReportDiagnostics.Severity.WARNING,
                "runtime_profile.primary_group_average_regressed",
                "Reduce runtime regression in group `" + candidateGroup.name() + "`",
                "The candidate trainer runtime group average is slower than the baseline.",
                groupActions(candidateGroup.name()),
                groupEvidence(baseline, candidate, baselineGroup, candidateGroup, regression)));
    }

    private static Optional<TrainingReportRecommendation> hotspotAverageRegression(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate,
            TrainingReportRuntimeRegressionSummary summary) {
        Optional<TrainingReportRuntimeRegressionSummary.Entry> maybeEntry = summary.primaryHotspotAverage()
                .filter(TrainingReportRuntimeRegressionSummary.Entry::regressed);
        if (maybeEntry.isEmpty()) {
            return Optional.empty();
        }
        Optional<TrainingReportRuntimeProfile.Hotspot> maybeCandidate = candidate.primaryHotspot();
        if (maybeCandidate.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Hotspot candidateHotspot = detailedHotspot(candidate, maybeCandidate.get());
        Optional<TrainingReportRuntimeProfile.Hotspot> maybeBaseline = baseline.hotspots().stream()
                .filter(hotspot -> hotspot.phase().equals(candidateHotspot.phase()))
                .findFirst();
        if (maybeBaseline.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Hotspot baselineHotspot = maybeBaseline.get();
        TrainingReportRuntimeRegressionSummary.Entry regression = maybeEntry.get();
        return Optional.of(new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.HIGH,
                category(candidateHotspot.phase()),
                TrainingReportDiagnostics.Severity.WARNING,
                "runtime_profile.primary_hotspot_average_regressed",
                "Reduce runtime regression in hotspot `" + candidateHotspot.phase() + "`",
                "The candidate trainer runtime hotspot average is slower than the baseline.",
                hotspotActions(candidateHotspot.phase()),
                hotspotEvidence(baseline, candidate, baselineHotspot, candidateHotspot, regression)));
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.Entry> groupAverageEntry(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate) {
        Optional<TrainingReportRuntimeProfile.Group> maybeCandidate = candidate.primaryGroup();
        if (maybeCandidate.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Group candidateGroup = detailedGroup(candidate, maybeCandidate.get());
        Optional<TrainingReportRuntimeProfile.Group> maybeBaseline = baseline.groups().stream()
                .filter(group -> group.name().equals(candidateGroup.name()))
                .findFirst();
        if (maybeBaseline.isEmpty()) {
            return Optional.empty();
        }
        return regression(
                candidateGroup.name(),
                "group",
                maybeBaseline.get().averageMillis(),
                candidateGroup.averageMillis(),
                MIN_GROUP_AVERAGE_REGRESSION_RATIO);
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.Entry> hotspotAverageEntry(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate) {
        Optional<TrainingReportRuntimeProfile.Hotspot> maybeCandidate = candidate.primaryHotspot();
        if (maybeCandidate.isEmpty()) {
            return Optional.empty();
        }
        TrainingReportRuntimeProfile.Hotspot candidateHotspot = detailedHotspot(candidate, maybeCandidate.get());
        Optional<TrainingReportRuntimeProfile.Hotspot> maybeBaseline = baseline.hotspots().stream()
                .filter(hotspot -> hotspot.phase().equals(candidateHotspot.phase()))
                .findFirst();
        if (maybeBaseline.isEmpty()) {
            return Optional.empty();
        }
        return regression(
                candidateHotspot.phase(),
                "hotspot",
                maybeBaseline.get().averageMillis(),
                candidateHotspot.averageMillis(),
                MIN_HOTSPOT_AVERAGE_REGRESSION_RATIO);
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.EfficiencyEntry> accountedWallTimeEntry(
            TrainingReportRuntimeEfficiency baseline,
            TrainingReportRuntimeEfficiency candidate) {
        if (baseline.accountedPercent().isEmpty() || candidate.accountedPercent().isEmpty()) {
            return Optional.empty();
        }
        double baselineValue = baseline.accountedPercent().orElseThrow();
        double candidateValue = candidate.accountedPercent().orElseThrow();
        return Optional.of(new TrainingReportRuntimeRegressionSummary.EfficiencyEntry(
                "accountedWallTime",
                "efficiency",
                "lower_is_worse",
                baselineValue,
                candidateValue,
                candidateValue - baselineValue,
                MIN_ACCOUNTED_WALL_TIME_DROP_PERCENT,
                "percent"));
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.EfficiencyEntry> wallClockOverheadEntry(
            TrainingReportRuntimeEfficiency baseline,
            TrainingReportRuntimeEfficiency candidate) {
        if (baseline.overheadPercent().isEmpty() || candidate.overheadPercent().isEmpty()) {
            return Optional.empty();
        }
        String key = candidate.overheadScope();
        double baselineValue = baseline.overheadPercent().orElseThrow();
        double candidateValue = candidate.overheadPercent().orElseThrow();
        return Optional.of(new TrainingReportRuntimeRegressionSummary.EfficiencyEntry(
                key,
                "wallClockOverhead",
                "higher_is_worse",
                baselineValue,
                candidateValue,
                candidateValue - baselineValue,
                MIN_WALL_CLOCK_OVERHEAD_INCREASE_PERCENT,
                "percent"));
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.EfficiencyEntry> dominantBottleneckEntry(
            TrainingReportRuntimeEfficiency baseline,
            TrainingReportRuntimeEfficiency candidate) {
        if (baseline.bottleneckPercent().isEmpty() || candidate.bottleneckPercent().isEmpty()) {
            return Optional.empty();
        }
        if (!baseline.bottleneck().equals(candidate.bottleneck())) {
            return Optional.empty();
        }
        double baselineValue = baseline.bottleneckPercent().orElseThrow();
        double candidateValue = candidate.bottleneckPercent().orElseThrow();
        return Optional.of(new TrainingReportRuntimeRegressionSummary.EfficiencyEntry(
                candidate.bottleneck(),
                "dominantBottleneck",
                "higher_is_worse",
                baselineValue,
                candidateValue,
                candidateValue - baselineValue,
                MIN_BOTTLENECK_INCREASE_PERCENT,
                "percent"));
    }

    private static Optional<TrainingReportRuntimeRegressionSummary.Entry> regression(
            String key,
            String kind,
            java.util.OptionalDouble baselineAverageMillis,
            java.util.OptionalDouble candidateAverageMillis,
            double thresholdRatio) {
        if (baselineAverageMillis.isEmpty() || candidateAverageMillis.isEmpty()) {
            return Optional.empty();
        }
        double baseline = baselineAverageMillis.orElseThrow();
        double candidate = candidateAverageMillis.orElseThrow();
        if (baseline <= 0.0 || candidate <= 0.0) {
            return Optional.empty();
        }
        double ratio = candidate / baseline;
        return Optional.of(new TrainingReportRuntimeRegressionSummary.Entry(
                key,
                kind,
                baseline,
                candidate,
                ratio,
                thresholdRatio));
    }

    private static TrainingReportRuntimeProfile.Group detailedGroup(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group primary) {
        return profile.groups().stream()
                .filter(group -> group.name().equals(primary.name()))
                .findFirst()
                .orElse(primary);
    }

    private static TrainingReportRuntimeProfile.Hotspot detailedHotspot(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Hotspot primary) {
        return profile.hotspots().stream()
                .filter(hotspot -> hotspot.phase().equals(primary.phase()))
                .findFirst()
                .orElse(primary);
    }

    private static TrainingReportRecommendation.Category category(String key) {
        if (key.startsWith("optimizer")) {
            return TrainingReportRecommendation.Category.OPTIMIZATION;
        }
        if (key.startsWith("input")) {
            return TrainingReportRecommendation.Category.DATA_HEALTH;
        }
        if (key.startsWith("validation")) {
            return TrainingReportRecommendation.Category.VALIDATION;
        }
        return TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
    }

    private static List<String> groupActions(String group) {
        return switch (group) {
            case "optimizer" -> List.of(
                    "Compare optimizer diagnostics, clipping, scheduler, and parameter update timings against the baseline.",
                    "Check whether candidate diagnostics are less sampled or scanning larger tensor sets.",
                    "Run a short optimizer-only profile before changing model architecture.");
            case "train" -> List.of(
                    "Compare train forward, backward, loss, metrics, and batch adaptation timing against the baseline.",
                    "Check whether candidate data preprocessing, dynamic shapes, or backend transfers changed.",
                    "Keep the baseline report alongside the candidate report until the regression source is isolated.");
            default -> List.of(
                    "Compare child phase timings against the baseline before changing unrelated trainer settings.",
                    "Run a deterministic short profile for both baseline and candidate.",
                    "Add a regression benchmark once the slower child phase is identified.");
        };
    }

    private static List<String> hotspotActions(String phase) {
        if (phase.startsWith("optimizer.")) {
            return List.of(
                    "Inspect optimizer-side timing changes for this phase before tuning model compute.",
                    "Check whether diagnostics, clipping, or parameter scans became more frequent.",
                    "Compare this hotspot with diagnostics sampling enabled and disabled.");
        }
        if (phase.startsWith("train.") || phase.startsWith("validation.")) {
            return List.of(
                    "Compare tensor shapes, batch size, and backend placement for this phase against the baseline.",
                    "Check whether data conversion or target preparation moved into the hot path.",
                    "Run a focused phase benchmark before changing optimizer settings.");
        }
        return List.of(
                "Compare this phase against the baseline with the same seed and data order.",
                "Inspect recent trainer changes that affect this phase directly.",
                "Add a focused benchmark once the regression is reproduced.");
    }

    private static List<String> bottleneckRegressionActions(String bottleneck) {
        return switch (bottleneck) {
            case "input" -> List.of(
                    "Compare baseline and candidate DataLoader prefetch, collation, decoding, and augmentation settings.",
                    "Inspect whether candidate input work moved into the synchronous batch path.",
                    "Add an input-pipeline regression benchmark with fixed seed and sample order.");
            case "optimizer" -> List.of(
                    "Compare optimizer diagnostics, clipping, scheduler, and parameter-scan settings against the baseline.",
                    "Sample expensive optimizer diagnostics when full tensor scans explain the regression.",
                    "Run an optimizer-only profile before changing model architecture.");
            case "validation" -> List.of(
                    "Compare validation cadence, batch count, metric detail, and preprocessing against the baseline.",
                    "Cache validation-only setup when the candidate regresses in validation bottleneck share.",
                    "Add a validation-only runtime regression benchmark.");
            default -> List.of(
                    "Compare child phase timings against the baseline before changing unrelated trainer settings.",
                    "Run a deterministic short profile for both baseline and candidate.",
                    "Add a focused benchmark once the bottleneck regression is reproduced.");
        };
    }

    private static Map<String, Object> groupEvidence(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate,
            TrainingReportRuntimeProfile.Group baselineGroup,
            TrainingReportRuntimeProfile.Group candidateGroup,
            TrainingReportRuntimeRegressionSummary.Entry regression) {
        Map<String, Object> evidence = regressionEvidence(regression);
        evidence.put("group", candidateGroup.name());
        evidence.put("baselineGroupCount", baseline.groupCount());
        evidence.put("candidateGroupCount", candidate.groupCount());
        baselineGroup.percentTotal().ifPresent(value -> evidence.put("baselinePercentTotal", value));
        candidateGroup.percentTotal().ifPresent(value -> evidence.put("candidatePercentTotal", value));
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> hotspotEvidence(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate,
            TrainingReportRuntimeProfile.Hotspot baselineHotspot,
            TrainingReportRuntimeProfile.Hotspot candidateHotspot,
            TrainingReportRuntimeRegressionSummary.Entry regression) {
        Map<String, Object> evidence = regressionEvidence(regression);
        evidence.put("phase", candidateHotspot.phase());
        evidence.put("baselineHotspotCount", baseline.hotspotCount());
        evidence.put("candidateHotspotCount", candidate.hotspotCount());
        baselineHotspot.percentTotal().ifPresent(value -> evidence.put("baselinePercentTotal", value));
        candidateHotspot.percentTotal().ifPresent(value -> evidence.put("candidatePercentTotal", value));
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> regressionEvidence(TrainingReportRuntimeRegressionSummary.Entry regression) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("baselineAverageMillis", regression.baselineAverageMillis());
        evidence.put("candidateAverageMillis", regression.candidateAverageMillis());
        evidence.put("ratio", regression.ratio());
        evidence.put("threshold", regression.threshold());
        return evidence;
    }

    private static Map<String, Object> efficiencyEvidence(
            TrainingReportRuntimeRegressionSummary.EfficiencyEntry regression) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("key", regression.key());
        evidence.put("kind", regression.kind());
        evidence.put("direction", regression.direction());
        evidence.put("baselineValue", regression.baselineValue());
        evidence.put("candidateValue", regression.candidateValue());
        evidence.put("delta", regression.delta());
        evidence.put("threshold", regression.threshold());
        evidence.put("unit", regression.unit());
        return Map.copyOf(evidence);
    }
}
