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
        return List.copyOf(recommendations);
    }

    static TrainingReportRuntimeRegressionSummary summary(
            TrainingReportRuntimeProfile baseline,
            TrainingReportRuntimeProfile candidate) {
        if (baseline == null || candidate == null || !baseline.available() || !candidate.available()) {
            return TrainingReportRuntimeRegressionSummary.empty();
        }
        return new TrainingReportRuntimeRegressionSummary(
                groupAverageEntry(baseline, candidate),
                hotspotAverageEntry(baseline, candidate));
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
}
