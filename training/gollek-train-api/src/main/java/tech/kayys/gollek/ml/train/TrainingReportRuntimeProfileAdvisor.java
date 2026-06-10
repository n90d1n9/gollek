package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds actionable guidance from trainer runtime-profile metadata.
 */
final class TrainingReportRuntimeProfileAdvisor {
    private TrainingReportRuntimeProfileAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        TrainingReportRuntimeProfile profile = TrainingReportReader.runtimeProfileView(report);
        if (!profile.available()) {
            return List.of();
        }
        List<TrainingReportRecommendation> recommendations = new ArrayList<>(3);
        profile.primaryHotspot()
                .filter(hotspot -> !hotspot.phase().isBlank())
                .map(hotspot -> hotspotRecommendation(report, profile, hotspot))
                .ifPresent(recommendations::add);
        profile.primaryGroup()
                .filter(group -> !group.name().isBlank())
                .map(group -> groupRecommendation(report, profile, group))
                .ifPresent(recommendations::add);
        recommendations.addAll(TrainingReportRuntimeProfileStabilityAdvisor.recommendations(profile));
        recommendations.addAll(TrainingReportRuntimeProfileBalanceAdvisor.recommendations(report));
        recommendations.addAll(TrainingReportRuntimeWallClockAdvisor.recommendations(profile));
        recommendations.addAll(TrainingReportRuntimeEfficiencyAdvisor.recommendations(
                TrainingReportRuntimeEfficiency.from(profile)));
        return List.copyOf(recommendations);
    }

    private static TrainingReportRecommendation hotspotRecommendation(
            Map<String, ?> report,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Hotspot hotspot) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                hotspotCategory(hotspot.phase()),
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.primary_hotspot",
                "Review trainer runtime hotspot `" + hotspot.phase() + "`",
                "The runtime profiler reports `" + hotspot.phase() + "` as the slowest measured trainer phase.",
                hotspotActions(report, hotspot.phase()),
                hotspotEvidence(report, profile, hotspot));
    }

    private static TrainingReportRecommendation groupRecommendation(
            Map<String, ?> report,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.MEDIUM,
                groupCategory(group.name()),
                TrainingReportDiagnostics.Severity.INFO,
                "runtime_profile.primary_group",
                "Review trainer runtime group `" + group.name() + "`",
                "The runtime profiler reports `" + group.name() + "` as the slowest measured trainer phase family.",
                groupActions(report, group.name()),
                groupEvidence(report, profile, group));
    }

    private static TrainingReportRecommendation.Category hotspotCategory(String phase) {
        if (phase.startsWith("optimizer.")) {
            return TrainingReportRecommendation.Category.OPTIMIZATION;
        }
        if (phase.startsWith("input.")) {
            return TrainingReportRecommendation.Category.DATA_HEALTH;
        }
        return TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
    }

    private static TrainingReportRecommendation.Category groupCategory(String group) {
        if ("optimizer".equals(group)) {
            return TrainingReportRecommendation.Category.OPTIMIZATION;
        }
        if ("input".equals(group)) {
            return TrainingReportRecommendation.Category.DATA_HEALTH;
        }
        return TrainingReportRecommendation.Category.TRAINING_DYNAMICS;
    }

    private static List<String> groupActions(Map<String, ?> report, String group) {
        return switch (group) {
            case "optimizer" -> List.of(
                    "Break down optimizer time by step, gradient diagnostics, clipping, and scheduler phases before changing model code.",
                    "Use sampled diagnostics for large models when optimizer-side tensor scans dominate the grouped runtime.",
                    "Compare optimizer group timing before and after optimizer, clipping, or diagnostics changes.");
            case "input" -> inputActions(
                    report,
                    "Inspect train and validation input iterator, hasNext, and next timings before tuning model math.",
                    "Enable or increase data-loader prefetch buffering when input next/hasNext time dominates the run.",
                    "Move expensive decoding, augmentation, dynamic padding, or target conversion out of the synchronous batch path when possible.");
            case "validation" -> List.of(
                    "Check validation frequency and batch count before tuning training-loop compute.",
                    "Cache validation-only preprocessing and metric state that does not change between batches.",
                    "Run a short validation-only profile to confirm whether forward, loss, or metrics dominate validation time.");
            case "train" -> List.of(
                    "Inspect train forward, backward, loss, and metric timings together before tuning optimizer settings.",
                    "Compare grouped train timing across batch sizes and backends to find scaling limits.",
                    "Move static preprocessing, target conversion, or metric setup outside the hot batch path when possible.");
            default -> List.of(
                    "Inspect the grouped runtime table before tuning unrelated trainer settings.",
                    "Compare this group across two short profiling runs to confirm the bottleneck is stable.",
                    "Add a focused benchmark once the slow operation inside the group is identified.");
        };
    }

    private static List<String> hotspotActions(Map<String, ?> report, String phase) {
        if (phase.startsWith("input.")) {
            return inputActions(
                    report,
                    "Enable `DataLoader.prefetch(...)` or increase the prefetch buffer when input iteration waits dominate training.",
                    "Inspect expensive work inside loader `iterator()`, `hasNext()`, or `next()` such as decoding, augmentation, dynamic padding, or target conversion.",
                    "Compare `input.train.*` and `input.validation.*` timings to decide whether the bottleneck is training data, validation data, or shared preprocessing.");
        }
        if (phase.startsWith("train.forward") || phase.startsWith("validation.forward")) {
            return List.of(
                    "Inspect model forward hotspots and tensor shapes before changing optimizer settings.",
                    "Try smaller batches or sequence lengths to confirm whether compute scales as expected.",
                    "Enable the fastest available backend for the model path and compare CPU/GPU profile deltas.");
        }
        if (phase.startsWith("train.backward")) {
            return List.of(
                    "Inspect autograd graph size and unnecessary retained tensors.",
                    "Use gradient accumulation only when it improves memory pressure enough to justify extra work.",
                    "Compare backward timing before and after freezing or pruning non-essential parameters.");
        }
        if (phase.startsWith("optimizer.gradient") || phase.startsWith("optimizer.parameter")) {
            TrainingReportParameterUpdateDiagnosticsPolicy policy = parameterUpdateDiagnosticsPolicy(report);
            if (parameterDiagnosticsHotspot(phase) && policy.enabled() && !policy.sampled()) {
                return List.of(
                        "Increase `parameterUpdateDiagnosticsIntervalSteps` for large-model runs so exact parameter-update scans are sampled instead of every optimizer step.",
                        "Keep every-step parameter-update diagnostics only for short debug runs or suspected optimizer corruption.",
                        "Compare runtime profile hotspots before and after sampling to confirm diagnostic scans are no longer dominant.");
            }
            if (parameterDiagnosticsHotspot(phase) && policy.sampled()) {
                return List.of(
                        "Parameter-update diagnostics are already sampled every " + policy.intervalSteps()
                                + " optimizer step(s); inspect remaining tensor diagnostics before increasing the interval again.",
                        "If this phase still dominates runtime, reduce other full-tensor diagnostics or run them only in focused debug sessions.",
                        "Compare clipped-gradient and parameter validation timings to find the next scan-heavy diagnostic.");
            }
            return List.of(
                    "Review expensive gradient or parameter diagnostics and keep full diagnostics for debug runs only.",
                    "Prefer sampled diagnostics for large models when full tensor scans dominate runtime.",
                    "Check whether clipping thresholds are forcing extra tensor passes every optimizer step.");
        }
        if (phase.startsWith("optimizer.step")) {
            return List.of(
                    "Inspect optimizer choice, parameter group count, and state size.",
                    "Compare SGD/AdamW-style optimizer timing on a small representative run.",
                    "Check whether backend placement causes parameter transfers before each optimizer step.");
        }
        if (phase.startsWith("train.loss") || phase.startsWith("validation.loss")) {
            return List.of(
                    "Inspect loss function tensor shapes and avoid repeated target conversions inside the loss.",
                    "Cache static class weights or masks outside the batch loop.",
                    "Compare fused and unfused loss implementations on a short profiling run.");
        }
        return List.of(
                "Inspect this phase in the runtime profile before tuning unrelated trainer settings.",
                "Run a short baseline with the same data and model to confirm the hotspot is stable.",
                "Add a focused benchmark once the slow operation is identified.");
    }

    private static List<String> inputActions(Map<String, ?> report, String... fallbackActions) {
        List<String> actions = new ArrayList<>();
        inputBottleneckAction(report).ifPresent(actions::add);
        actions.addAll(List.of(fallbackActions));
        return List.copyOf(actions);
    }

    private static java.util.Optional<String> inputBottleneckAction(Map<String, ?> report) {
        Map<String, Object> evidence = TrainingReportRuntimeInputProfile.fromMetadata(TrainingReportReader.metadata(report))
                .toEvidenceMap();
        Object scope = evidence.get("dominantInputScope");
        Object stage = evidence.get("dominantInputStage");
        Object scopeTotal = evidence.get("dominantInputScopeTotalMillis");
        Object stageTotal = evidence.get("dominantInputStageTotalMillis");
        if (!(scope instanceof String scopeName) || !(stage instanceof String stageName)) {
            return java.util.Optional.empty();
        }
        StringBuilder action = new StringBuilder("Prioritize the `")
                .append(scopeName)
                .append("` input loader `")
                .append(stageName)
                .append("()` path");
        if (scopeTotal instanceof Number scopeMillis && stageTotal instanceof Number stageMillis) {
            action.append(" (")
                    .append(formatMillis(stageMillis.doubleValue()))
                    .append(" ms of ")
                    .append(formatMillis(scopeMillis.doubleValue()))
                    .append(" ms measured input time)");
        }
        action.append(" before tuning unrelated trainer compute.");
        return java.util.Optional.of(action.toString());
    }

    private static String formatMillis(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
    }

    private static Map<String, Object> hotspotEvidence(
            Map<String, ?> report,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Hotspot hotspot) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("phase", hotspot.phase());
        evidence.put("hotspotCount", profile.hotspotCount());
        hotspot.count().ifPresent(value -> evidence.put("count", value));
        hotspot.totalMillis().ifPresent(value -> evidence.put("totalMillis", value));
        hotspot.percentTotal().ifPresent(value -> evidence.put("percentTotal", value));
        hotspot.averageMillis().ifPresent(value -> evidence.put("averageMillis", value));
        hotspot.minMillis().ifPresent(value -> evidence.put("minMillis", value));
        hotspot.maxMillis().ifPresent(value -> evidence.put("maxMillis", value));
        hotspot.lastMillis().ifPresent(value -> evidence.put("lastMillis", value));
        hotspot.stddevMillis().ifPresent(value -> evidence.put("stddevMillis", value));
        if (parameterDiagnosticsHotspot(hotspot.phase())) {
            TrainingReportParameterUpdateDiagnosticsPolicy policy = parameterUpdateDiagnosticsPolicy(report);
            evidence.put("parameterUpdateDiagnosticsEnabled", policy.enabled());
            evidence.put("parameterUpdateDiagnosticsSampled", policy.sampled());
            evidence.put("parameterUpdateDiagnosticsIntervalSteps", policy.intervalSteps());
        }
        putInputProfileEvidence(report, hotspot.phase(), evidence);
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> groupEvidence(
            Map<String, ?> report,
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeProfile.Group group) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("group", group.name());
        evidence.put("groupCount", profile.groupCount());
        group.count().ifPresent(value -> evidence.put("count", value));
        group.totalMillis().ifPresent(value -> evidence.put("totalMillis", value));
        group.percentTotal().ifPresent(value -> evidence.put("percentTotal", value));
        group.averageMillis().ifPresent(value -> evidence.put("averageMillis", value));
        group.minMillis().ifPresent(value -> evidence.put("minMillis", value));
        group.maxMillis().ifPresent(value -> evidence.put("maxMillis", value));
        group.lastMillis().ifPresent(value -> evidence.put("lastMillis", value));
        group.stddevMillis().ifPresent(value -> evidence.put("stddevMillis", value));
        putInputProfileEvidence(report, group.name(), evidence);
        return Map.copyOf(evidence);
    }

    private static void putInputProfileEvidence(
            Map<String, ?> report,
            String phaseOrGroup,
            Map<String, Object> evidence) {
        if (phaseOrGroup == null || !phaseOrGroup.startsWith("input")) {
            return;
        }
        evidence.putAll(TrainingReportRuntimeInputProfile.fromMetadata(TrainingReportReader.metadata(report))
                .toEvidenceMap());
    }

    private static boolean parameterDiagnosticsHotspot(String phase) {
        return phase.startsWith("optimizer.parameter");
    }

    private static TrainingReportParameterUpdateDiagnosticsPolicy parameterUpdateDiagnosticsPolicy(
            Map<String, ?> report) {
        return TrainingReportReader.parameterUpdateDiagnosticsPolicyView(report);
    }
}
